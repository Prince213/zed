use crate::{
    DebugEvent, EditPredictionFinishedDebugEvent, EditPredictionId, EditPredictionModelInput,
    EditPredictionStartedDebugEvent, EditPredictionStore, buffer_path_with_id_fallback,
    prediction::EditPredictionResult,
    zeta::{compute_edits_and_cursor_position, zeta2_prompt_input},
};
use anyhow::{Context as _, Result};
use cloud_llm_client::EditPredictionRejectReason;
use futures::AsyncReadExt as _;
use gpui::{
    App, AppContext as _, Context, Entity, Global, SharedString, Task,
    http_client::{self, AsyncBody, HttpClient, Method},
};
use language::{Buffer, BufferSnapshot, OffsetRangeExt as _, ToOffset as _};
use language_model::ApiKeyState;
use open_ai::{
    Role,
    responses::{
        Request, ResponseInputContent, ResponseInputItem, ResponseMessageItem, ResponseOutputItem,
        ResponseSummary,
    },
};
use std::{ops::Range, sync::Arc};
use text::Anchor;
use zeta_prompt::{ParsedOutput, ZetaFormat, format_zeta_prompt, get_prefill};

pub fn open_ai_api_url(cx: &App) -> SharedString {
    language::language_settings::all_language_settings(None, cx)
        .edit_predictions
        .open_ai
        .as_ref()
        .map(|settings| settings.api_url.clone())
        .unwrap_or_else(|| open_ai::OPEN_AI_API_URL.into())
        .into()
}

struct GlobalOpenAiApiKey(Entity<ApiKeyState>);

impl Global for GlobalOpenAiApiKey {}

pub fn open_ai_api_token(cx: &mut App) -> Entity<ApiKeyState> {
    if let Some(global) = cx.try_global::<GlobalOpenAiApiKey>() {
        return global.0.clone();
    }

    let entity =
        cx.new(|cx| language_models::provider::open_ai::open_ai_api_key_state(open_ai_api_url(cx)));
    cx.set_global(GlobalOpenAiApiKey(entity.clone()));
    entity
}

pub fn load_open_ai_api_token(cx: &mut App) -> Task<Result<(), language_model::AuthenticateError>> {
    let credentials_provider = zed_credentials_provider::global(cx);
    let api_url = open_ai_api_url(cx);
    open_ai_api_token(cx).update(cx, |key_state, cx| {
        key_state.load_if_needed(api_url, |s| s, credentials_provider, cx)
    })
}

pub fn load_open_ai_api_key_if_needed(cx: &mut App) -> Option<Arc<str>> {
    _ = load_open_ai_api_token(cx);
    let url = open_ai_api_url(cx);
    open_ai_api_token(cx).read(cx).key(&url)
}

struct OpenAiResponsesOutput {
    request_id: String,
    buffer: Entity<Buffer>,
    snapshot: BufferSnapshot,
    prompt_input: zeta_prompt::ZetaPromptInput,
    parsed_output: Option<ParsedOutput>,
    full_context_offset_range: Range<usize>,
}

pub(crate) fn request_prediction(
    EditPredictionModelInput {
        buffer,
        snapshot,
        position,
        related_files,
        events,
        debug_tx,
        diagnostic_search_range,
        is_open_source,
        can_collect_data,
        ..
    }: EditPredictionModelInput,
    cx: &mut Context<EditPredictionStore>,
) -> Task<Result<Option<EditPredictionResult>>> {
    let Some(settings) =
        language::language_settings::all_language_settings(None, cx)
            .edit_predictions
            .open_ai
            .clone()
    else {
        return Task::ready(Ok(None));
    };

    let Some(api_key) = load_open_ai_api_key_if_needed(cx) else {
        return Task::ready(Ok(None));
    };

    let http_client = cx.http_client();
    let request_start = cx.background_executor().now();
    let excerpt_path = buffer_path_with_id_fallback(snapshot.file(), &snapshot.text, cx);
    let cursor_offset = position.to_offset(&snapshot);
    let active_buffer = buffer.clone();
    let zeta_format = ZetaFormat::default();

    let result = cx.background_spawn(async move {
        let (full_context_offset_range, prompt_input) = zeta2_prompt_input(
            &snapshot,
            related_files,
            events,
            diagnostic_search_range,
            excerpt_path,
            cursor_offset,
            is_open_source,
            can_collect_data,
            None,
        );

        let Some(prompt) = format_zeta_prompt(&prompt_input, zeta_format) else {
            return anyhow::Ok(None);
        };

        let prefill = get_prefill(&prompt_input, zeta_format);
        let prompt = format!("{prompt}{prefill}");

        if let Some(debug_tx) = &debug_tx {
            debug_tx
                .unbounded_send(DebugEvent::EditPredictionStarted(
                    EditPredictionStartedDebugEvent {
                        buffer: active_buffer.downgrade(),
                        prompt: Some(prompt.clone()),
                        position,
                    },
                ))
                .ok();
        }

        let request = build_responses_request(&settings, prompt);
        let response =
            send_responses_request(&settings.api_url, &api_key, request, &http_client).await?;
        let request_id = response
            .id
            .clone()
            .context("OpenAI Responses response missing id")?;
        let response_text = output_text_from_response(&response)?;

        if let Some(debug_tx) = &debug_tx {
            debug_tx
                .unbounded_send(DebugEvent::EditPredictionFinished(
                    EditPredictionFinishedDebugEvent {
                        buffer: active_buffer.downgrade(),
                        model_output: response_text.clone(),
                        position,
                    },
                ))
                .ok();
        }

        let parsed_output = if let Some(response_text) = response_text {
            let output = format!("{prefill}{response_text}");
            Some(zeta_prompt::parse_zeta2_model_output(
                &output,
                zeta_format,
                &prompt_input,
            )?)
        } else {
            None
        };

        anyhow::Ok(Some(OpenAiResponsesOutput {
            request_id,
            buffer,
            snapshot,
            prompt_input,
            parsed_output,
            full_context_offset_range,
        }))
    });

    cx.spawn(async move |_this, cx| {
        let Some(output) = result.await.context("OpenAI edit prediction failed")? else {
            return Ok(None);
        };

        let request_duration = cx.background_executor().now() - request_start;
        let id = EditPredictionId(output.request_id.into());

        let Some(ParsedOutput {
            new_editable_region: mut output_text,
            range_in_excerpt: editable_range_in_excerpt,
            cursor_offset_in_new_editable_region: cursor_offset_in_output,
        }) = output.parsed_output
        else {
            return Ok(Some(EditPredictionResult {
                id,
                prediction: Err(EditPredictionRejectReason::Empty),
                model_version: None,
                e2e_latency: request_duration,
            }));
        };

        let editable_range_in_buffer = editable_range_in_excerpt.start
            + output.full_context_offset_range.start
            ..editable_range_in_excerpt.end + output.full_context_offset_range.start;
        let mut old_text = output
            .snapshot
            .text_for_range(editable_range_in_buffer.clone())
            .collect::<String>();

        if !output_text.is_empty() && !output_text.ends_with('\n') {
            output_text.push('\n');
        }
        if !old_text.is_empty() && !old_text.ends_with('\n') {
            old_text.push('\n');
        }

        let (edits, cursor_position) = compute_edits_and_cursor_position(
            old_text,
            &output_text,
            editable_range_in_buffer.start,
            cursor_offset_in_output,
            &output.snapshot,
        );

        Ok(Some(
            EditPredictionResult::new(
                id,
                &output.buffer,
                &output.snapshot,
                edits.into(),
                cursor_position,
                Some(output.snapshot.anchor_range_inside(editable_range_in_buffer)),
                output.prompt_input,
                None,
                request_duration,
                cx,
            )
            .await,
        ))
    })
}

pub(crate) fn build_responses_request(
    settings: &language::language_settings::OpenAiEditPredictionSettings,
    prompt: String,
) -> Request {
    Request {
        model: settings.model.clone(),
        instructions: None,
        input: vec![ResponseInputItem::Message(ResponseMessageItem {
            role: Role::User,
            content: vec![ResponseInputContent::Text { text: prompt }],
            phase: None,
        })],
        include: Vec::new(),
        stream: false,
        temperature: None,
        top_p: None,
        max_output_tokens: Some(settings.max_output_tokens.into()),
        parallel_tool_calls: None,
        tool_choice: None,
        tools: Vec::new(),
        prompt_cache_key: None,
        reasoning: None,
        store: Some(false),
        service_tier: None,
    }
}

pub(crate) async fn send_responses_request(
    api_url: &str,
    api_key: &str,
    request: Request,
    http_client: &Arc<dyn HttpClient>,
) -> Result<ResponseSummary> {
    let uri = format!("{}/responses", api_url.trim_end_matches('/'));
    let request_body = serde_json::to_string(&request)?;
    let request = http_client::Request::builder()
        .uri(uri)
        .header("Content-Type", "application/json")
        .header("Authorization", format!("Bearer {}", api_key.trim()))
        .method(Method::POST)
        .body(AsyncBody::from(request_body))
        .context("Failed to create OpenAI Responses request")?;

    let mut response = http_client
        .send(request)
        .await
        .context("Failed to send OpenAI Responses request")?;

    let mut body = String::new();
    response
        .body_mut()
        .read_to_string(&mut body)
        .await
        .context("Failed to read OpenAI Responses response body")?;

    if !response.status().is_success() {
        anyhow::bail!(
            "OpenAI Responses request failed with status: {:?}\nBody: {}",
            response.status(),
            body,
        );
    }

    serde_json::from_str(&body).context("Failed to parse OpenAI Responses response")
}

pub(crate) fn output_text_from_response(response: &ResponseSummary) -> Result<Option<String>> {
    match response.status.as_deref() {
        Some("failed") => {
            let message = response
                .error
                .as_ref()
                .map(|error| error.message.as_str())
                .unwrap_or("unknown error");
            anyhow::bail!("OpenAI Responses request failed: {message}");
        }
        Some("incomplete") => {
            let reason = response
                .incomplete_details
                .as_ref()
                .and_then(|details| details.reason.as_deref())
                .unwrap_or("unknown reason");
            anyhow::bail!("OpenAI Responses request incomplete: {reason}");
        }
        _ => {}
    }

    let mut output_text = String::new();
    for item in &response.output {
        let ResponseOutputItem::Message(message) = item else {
            continue;
        };

        for content in &message.content {
            if content.get("type").and_then(|value| value.as_str()) != Some("output_text") {
                continue;
            }

            let Some(text) = content.get("text").and_then(|value| value.as_str()) else {
                continue;
            };
            output_text.push_str(text);
        }
    }

    if output_text.is_empty() {
        Ok(None)
    } else {
        Ok(Some(output_text))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::AsyncReadExt as _;
    use gpui::http_client::{FakeHttpClient, Response};
    use pretty_assertions::assert_eq;
    use serde_json::json;

    fn test_settings() -> language::language_settings::OpenAiEditPredictionSettings {
        language::language_settings::OpenAiEditPredictionSettings {
            api_url: "https://api.openai.test/v1".into(),
            model: "gpt-test".to_string(),
            max_output_tokens: 512,
        }
    }

    fn response_summary(value: serde_json::Value) -> ResponseSummary {
        serde_json::from_value(value).unwrap()
    }

    #[test]
    fn builds_responses_request() {
        let request = build_responses_request(&test_settings(), "predict this".to_string());
        let value = serde_json::to_value(&request).unwrap();

        assert_eq!(value["model"], "gpt-test");
        assert_eq!(value["max_output_tokens"], 512);
        assert_eq!(value["store"], false);
        assert_eq!(value["stream"], false);
        assert_eq!(value["input"][0]["type"], "message");
        assert_eq!(value["input"][0]["role"], "user");
        assert_eq!(value["input"][0]["content"][0]["type"], "input_text");
        assert_eq!(value["input"][0]["content"][0]["text"], "predict this");
    }

    #[gpui::test]
    async fn sends_responses_request_to_base_url() {
        let http_client = FakeHttpClient::create(|mut request| async move {
            assert_eq!(request.method(), Method::POST);
            assert_eq!(
                request.uri().to_string(),
                "https://api.openai.test/v1/responses"
            );
            assert_eq!(
                request
                    .headers()
                    .get("Authorization")
                    .unwrap()
                    .to_str()
                    .unwrap(),
                "Bearer sk-test"
            );

            let mut body = String::new();
            request.body_mut().read_to_string(&mut body).await.unwrap();
            let value: serde_json::Value = serde_json::from_str(&body).unwrap();
            assert_eq!(value["model"], "gpt-test");
            assert_eq!(value["max_output_tokens"], 512);
            assert_eq!(value["store"], false);
            assert_eq!(value["input"][0]["content"][0]["text"], "predict this");

            Ok(Response::builder()
                .status(200)
                .body(
                    json!({
                        "id": "resp_123",
                        "status": "completed",
                        "output": []
                    })
                    .to_string()
                    .into(),
                )
                .unwrap())
        });
        let http_client: Arc<dyn HttpClient> = http_client;

        let response = send_responses_request(
            "https://api.openai.test/v1",
            "sk-test",
            build_responses_request(&test_settings(), "predict this".to_string()),
            &http_client,
        )
        .await
        .unwrap();

        assert_eq!(response.id.as_deref(), Some("resp_123"));
    }

    #[test]
    fn extracts_output_text_from_response() {
        let response = response_summary(json!({
            "id": "resp_123",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "replacement",
                            "annotations": []
                        }
                    ]
                }
            ]
        }));

        assert_eq!(
            output_text_from_response(&response).unwrap(),
            Some("replacement".to_string())
        );
    }

    #[test]
    fn empty_output_is_no_prediction() {
        let response = response_summary(json!({
            "id": "resp_123",
            "status": "completed",
            "output": []
        }));

        assert_eq!(output_text_from_response(&response).unwrap(), None);
    }

    #[test]
    fn malformed_or_non_message_output_is_no_prediction() {
        let response = response_summary(json!({
            "id": "resp_123",
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "name": "tool",
                    "arguments": "{}"
                },
                {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "content": [
                        { "type": "refusal", "refusal": "no" },
                        { "type": "output_text", "text": 1 }
                    ]
                }
            ]
        }));

        assert_eq!(output_text_from_response(&response).unwrap(), None);
    }

    #[test]
    fn failed_response_is_error() {
        let response = response_summary(json!({
            "id": "resp_123",
            "status": "failed",
            "error": {
                "message": "model unavailable"
            },
            "output": []
        }));

        assert!(
            output_text_from_response(&response)
                .unwrap_err()
                .to_string()
                .contains("model unavailable")
        );
    }

    #[test]
    fn incomplete_response_is_error() {
        let response = response_summary(json!({
            "id": "resp_123",
            "status": "incomplete",
            "incomplete_details": {
                "reason": "max_output_tokens"
            },
            "output": []
        }));

        assert!(
            output_text_from_response(&response)
                .unwrap_err()
                .to_string()
                .contains("max_output_tokens")
        );
    }
}
