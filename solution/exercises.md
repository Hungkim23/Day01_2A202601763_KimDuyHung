# K3 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 9h00–13h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Quy luật nhận thấy qua bốn phản hồi: Khi temperature tăng từ 0.0 đến 1.5, câu trả lời chuyển dần từ tính xác định, chuẩn mực sang tính đa dạng và ngẫu nhiên cao. Ở mức 0.0 và 0.5, model trả về các sự thật phổ biến (như lịch sử, địa lý) với cấu trúc rất ổn định và nhất quán giữa các lần gọi. Ở mức 1.0 và 1.5, văn phong trở nên sáng tạo hơn, từ ngữ phong phú hơn nhưng từ 1.5 bắt đầu xuất hiện nguy cơ lan man hoặc đưa thông tin kém chính xác (hallucination).

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Mức temperature đề xuất: 0.0 đến 0.2. Lý do: Chatbot hỗ trợ khách hàng yêu cầu độ chính xác, tính thống nhất và độ tin cậy tuyệt đối về thông tin sản phẩm, chính sách hay giá cả. Đặt temperature thấp giúp model hạn chế tối đa việc "sáng tạo" tự do dẫn đến tư vấn sai sự thật cho khách hàng.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> So sánh chi phí:Giá token đầu ra (output) của GPT-4o là $0.010 / 1K tokens, trong khi GPT-4o-mini là $0.0006 / 1K tokens. GPT-4o đắt hơn GPT-4o-mini khoảng 16.67 lần cho phần output token.Với workload $10.000 * 3 * 350 = 10,5$ triệu output token/ngày, dùng GPT-4o tốn khoảng $105/ngày trong khi GPT-4o-mini chỉ tốn khoảng $6.3/ngày.Trường hợp nên dùng GPT-4o: Phân tích tài chính/pháp lý phức tạp, trích xuất dữ liệu từ hình ảnh hoặc xử lý các tác vụ lập luận logic chuyên sâu đòi hỏi độ chính xác cao.Trường hợp nên dùng GPT-4o-mini: Chatbot FAQ trả lời thắc mắc thường gặp, phân loại ý định người dùng (intent classification) hoặc tóm tắt văn bản ngắn.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> So sánh sự khác nhau: Phản hồi từ persona "giáo viên tiểu học" có độ dài ngắn, từ vựng đơn giản, sử dụng hình ảnh ẩn dụ gần gũi (như một cuốn sổ nhật ký truyền tay nhau). Ngược lại, persona "chuyên gia tài chính" đưa ra phản hồi dài hơn, sử dụng nhiều thuật ngữ chuyên ngành (sổ cái phân tán, mã hóa, cơ chế đồng thuận, tính phi tập trung). System prompt đóng vai trò như một chỉ dẫn ngữ cảnh định hình trực tiếp văn phong, độ sâu kiến thức và góc nhìn phản hồi của model.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Chênh lệch và nguyên nhân: Với đoạn văn tiếng Việt ~100 từ, ước tính số từ / 0.75 ra khoảng 133 token, nhưng đếm bằng tiktoken thực tế thường dao động khoảng 220–250 token (chênh lệch cao hơn khoảng 65%–85%). Tiếng Việt tốn nhiều token hơn tiếng Anh vì thuật toán mã hóa (Byte-Pair Encoding) của OpenAI được tối ưu hóa chủ yếu trên dữ liệu tiếng Anh. Các ký tự có dấu (ê, ơ, ư, dấu thanh) và cấu trúc từ ghép tiếng Việt thường bị tách thành nhiều sub-word hoặc byte token lẻ thay vì giữ nguyên thành 1 token hoàn chỉnh như tiếng Anh.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)

> Streaming quan trọng nhất trong các ứng dụng tương tác thời gian thực như chatbot CLI hoặc ứng dụng chat web, nơi phản hồi dài có thể mất vài giây đến hàng chục giây để sinh xong. Việc stream từng token giúp tối ưu chỉ số Time-To-First-Token (TTFT), mang lại cảm giác hệ thống phản hồi tức thì và nâng cao trải nghiệm người dùng. Ngược lại, non-streaming phù hợp hơn cho các tác vụ xử lý ngầm (background jobs) như trích xuất dữ liệu cấu trúc dạng JSON, phân tích cảm xúc văn bản, tổng hợp báo cáo tự động hoặc khi gọi API trực tiếp giữa hai hệ thống backend (service-to-service).

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**

> Exponential backoff giúp tăng dần khoảng thời gian chờ giữa các lần thử lại ($0.1s -> 0.2s -> 0.4s...), tạo khoảng nghỉ cho hệ thống server đang bị quá tải hoặc nghẽn mạng có thời gian tự phục hồi. Nếu hàng nghìn client cùng retry với một khoảng delay cố định (ví dụ cùng chờ 1 giây), toàn bộ các client sẽ đồng loạt gửi request lại đúng cùng một thời điểm ở chu kỳ tiếp theo. Điều này gây ra hiện tượng Thundering Herd Problem, khiến server tiếp tục bị ngập request và sập liên tục mà không thể khôi phục lại trạng thái bình thường.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> System Prompt: "Bạn là Trợ lý Lập trình Python & AI thân thiện. Hãy trả lời ngắn gọn dưới 3 câu, đi thẳng vào trọng tâm vấn đề bằng tiếng Việt và ưu tiên đính kèm ví dụ code minh họa tối giản khi giải thích kỹ thuật.". Giải thích lựa chọn từ ngữ: "trả lời ngắn gọn dưới 3 câu": Giúp kiểm soát số lượng token đầu ra, giảm latency và tiết kiệm chi phí API. "ưu tiên đính kèm ví dụ code minh họa tối giản": Đảm bảo câu trả lời mang tính thực hành cao, giúp sinh viên nắm bắt ngay cách triển khai mà không bị rối bởi lý thuyết dài dòng.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất: Trợ lý hiện tại bị giới hạn bộ nhớ cứng ở 3 lượt hội thoại gần nhất (history[-6:]), dẫn đến việc "quên" hoàn toàn ngữ cảnh hoặc các yêu cầu quan trọng mà người dùng đã đề cập từ đầu phiên chat nếu cuộc trò chuyện kéo dài.Đề xuất cải thiện: Triển khai kỹ thuật Conversation Summarization (Tóm tắt hội thoại). Khi lịch sử vượt quá 3 lượt, hệ thống sẽ tự động gọi một model nhỏ hơn (như gpt-4o-mini) để tóm tắt các lượt thoại cũ thành 1–2 câu ngắn gọn và chèn vào ngay sau system prompt. Cách này giúp trợ lý duy trì được bộ nhớ dài hạn mà tổng số lượng token gửi lên API vẫn được giữ ở mức thấp.


---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/` và zip theo hướng dẫn README
