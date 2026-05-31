# Assistant Integration Notes

These notes come from testing PMB inside a long-running local voice assistant, where memory is used across many chats, voice commands, reminders, and follow-up questions.

## Practical recommendations

- Keep user-authored content and assistant-authored content clearly separated before writing memories. The assistant answer can be useful context, but it should not be stored as a user fact by accident.
- Store stable facts as normalized statements, not raw chat quotes. For example, `The user has a friend named Alexey` is easier to reuse than `You know, I have a friend Alexey`.
- Preserve source metadata for every memory: role, chat/session id, timestamp, and whether the memory came from an explicit request such as `remember this` or from automatic extraction.
- Treat automatic extraction more conservatively than explicit memory commands. A spoken assistant may hear noisy or incomplete phrases, so confidence and source tracking matter.
- For follow-up questions, retrieval should prefer facts from the same user identity and then fall back to broader conversation summaries.
- Conversation summaries are useful, but they should not replace atomic facts. Summaries help the assistant remember the story of a dialogue; facts help it answer direct questions reliably.

## Common integration pitfalls

- Mixing assistant replies into factual memory can make the assistant later recall its own generated text as if the user had said it.
- Saving raw multilingual user phrases without normalization can make later retrieval feel inconsistent in bilingual assistants.
- Relying only on current chat context breaks cross-chat memory. Long-lived assistants need user-scoped retrieval independent of the active chat window.
- Returning a bare memory acknowledgement for every stored fact can feel unnatural in conversation. In voice assistants, it is often better to silently store low-risk facts and only acknowledge explicit `remember` requests.

## Useful behavior for voice assistants

A good assistant integration usually needs two layers around PMB:

1. A memory decision layer that decides whether a message contains durable information worth remembering.
2. A memory shaping layer that converts the message into short, stable, user-scoped facts before writing them.

PMB provides the storage and retrieval foundation; the host assistant should still own identity resolution, routing, safety rules, and when to write memories.