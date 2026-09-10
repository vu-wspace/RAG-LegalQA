# RAG-LegalQA — Hệ thống Hỏi-Đáp Pháp luật tiếng Việt bằng RAG

Hệ thống hỏi đáp pháp luật tiếng Việt (Legal Question Answering) ứng dụng kiến trúc **Retrieval-Augmented Generation (RAG)**: truy xuất các điều khoản pháp luật liên quan từ kho văn bản hành chính, sau đó dùng mô hình ngôn ngữ lớn (LLM) để sinh câu trả lời bằng văn xuôi, có trích dẫn căn cứ pháp lý.

Dự án được xây dựng cho **Task 2 – LegalQA** của cuộc thi **UIT Data Science Challenge 2026**.

## Bối cảnh

Trong quá trình chuyển đổi số tại Việt Nam, khối lượng văn bản pháp luật ngày càng lớn và phức tạp, khiến người dân, doanh nghiệp và cơ quan quản lý gặp khó khăn khi tra cứu, đối chiếu quy định phù hợp. Task 2 của cuộc thi yêu cầu: với mỗi câu hỏi đầu vào, hệ thống phải sinh câu trả lời bằng văn xuôi dựa trên nội dung pháp luật, đúng ngữ cảnh và văn phong pháp lý như các chuyên gia biên soạn. Dữ liệu dùng chung cho cả 2 tác vụ của cuộc thi gồm khoảng **8.500 văn bản hành chính** do Ban Tổ chức cung cấp.

## Kiến trúc hệ thống

```
                         INDEXING (offline)
Văn bản pháp luật (passages)
      │
      ▼
Domain-aware Chunking   — tách theo "Điều X" bằng regex, sau đó
                           RecursiveCharacterTextSplitter (chunk=1200, overlap=200)
      │
      ▼
Embedding (BGE-M3)      — dense vector 1024 chiều, max_length=8192
      │
      ▼
FAISS IndexFlatIP       — vector store, cosine similarity (normalize_L2)


                         QUERY (online)
Câu hỏi
      │
      ▼
Dense Retrieval (BGE-M3, k=20)
      │
      ▼
Cross-Encoder Reranking (bge-reranker-v2-m3, top_n=5)
      │
      ▼
Context Assembly       — ghép các đoạn liên quan thành ngữ cảnh
      │
      ▼
Generation (Qwen2.5-3B-Instruct)  — sinh câu trả lời tiếng Việt,
                                     có trích dẫn Điều/Khoản
      │
      ▼
Câu trả lời văn xuôi (submission.json)
```

### 1. Indexing

- **Chunking theo cấu trúc pháp lý**: dùng regex nhận diện ranh giới `Điều <số>` để tách văn bản theo đơn vị điều luật trước, tránh cắt ngang một điều thành nhiều mảnh không liên quan. Với điều luật quá dài mới tiếp tục chia nhỏ bằng `RecursiveCharacterTextSplitter` (ưu tiên tách theo `Khoản`, đoạn, câu).
- **Embedding**: `BAAI/bge-m3` — mô hình embedding đa ngôn ngữ hỗ trợ tốt tiếng Việt, `max_length=8192` phù hợp với đoạn văn bản pháp luật dài.
- **Vector store**: FAISS `IndexFlatIP` với vector đã chuẩn hoá L2 (tương đương cosine similarity).

### 2. Query pipeline

- **Retrieval 2 tầng**: lấy `k=20` ứng viên bằng dense retrieval, sau đó rerank bằng cross-encoder `bge-reranker-v2-m3` để chọn `top_n=5` đoạn liên quan nhất — giảm nhiễu so với chỉ dùng dense retrieval đơn thuần.
- **Xử lý theo batch có khả năng resume**: khi retrieve context hoặc sinh câu trả lời, hệ thống kiểm tra kết quả đã có (theo `question_id`) để bỏ qua, chỉ xử lý phần còn thiếu — quan trọng khi chạy trên tài nguyên giới hạn (Kaggle) với hàng nghìn câu hỏi.
- **Generation**: `Qwen/Qwen2.5-3B-Instruct`, sinh theo batch với `tokenizer.apply_chat_template`, `do_sample=False` (deterministic) để đảm bảo tính nhất quán cho bài nộp.
- **Chống hallucination bằng prompt engineering**: system prompt (tiếng Việt) quy định rõ:
  - Chỉ trả lời dựa trên ngữ cảnh được cung cấp, không dùng kiến thức ngoài.
  - Nếu không có câu trả lời rõ ràng trong ngữ cảnh → trả lời cố định "Tôi không biết dựa trên các tài liệu được cung cấp."
  - Bắt buộc trả lời hoàn toàn bằng tiếng Việt, không lẫn tiếng Anh, không lặp lại câu hỏi.
  - Trích dẫn số Điều/Khoản tự nhiên trong câu trả lời, đúng văn phong pháp lý.
- **Xử lý lỗi hết bộ nhớ GPU (OOM)**: khi generate theo batch bị OOM, tự động fallback xử lý từng câu một; nếu vẫn OOM thì trả về câu mặc định thay vì crash toàn bộ pipeline.

## Dữ liệu

- ~8.500 văn bản hành chính (do Ban Tổ chức UIT Data Science Challenge 2026 cung cấp).
- Bộ câu hỏi Task 2 (LegalQA) kèm các đoạn ngữ cảnh liên quan (`selected-contexts`) do chuyên gia pháp lý gắn nhãn.

## Công nghệ sử dụng

`Python` · `LangChain` (Document, text splitters) · `BGE-M3` (embedding) · `FAISS` (vector search) · `bge-reranker-v2-m3` (cross-encoder reranking, `sentence-transformers`) · `Qwen2.5-3B-Instruct` (LLM sinh câu trả lời) · `PyTorch` / `Transformers`

## Cấu trúc mã nguồn (theo notebook)

| Phần | Chức năng |
|---|---|
| `load_questions`, `load_answers` | Nạp câu hỏi và các đoạn văn bản pháp luật (passages) |
| `split_doc`, `chunking` | Domain-aware chunking theo Điều + recursive splitting |
| `bge_m3`, `embeddings` | Sinh dense embedding cho các chunk |
| `vectorstore` | Xây dựng và lưu FAISS index |
| `retriever` (`retrieve`, `retrieve_batch`) | Truy xuất top-k ứng viên theo câu hỏi |
| `get_reranker`, `rerank`, `rerank_batch` | Rerank ứng viên bằng cross-encoder |
| `contexts` | Pipeline retrieve context cho toàn bộ câu hỏi, có resume |
| `load_model`, `generate`, `generate_batch` | Nạp LLM và sinh câu trả lời theo batch |
| `make_submission` | Sinh file `submission.json` cuối cùng, có resume + xử lý OOM |

## Hạn chế & hướng cải thiện

- **Chưa có bộ đánh giá định lượng** (Recall@k cho retrieval; F1/ROUGE hoặc LLM-as-judge cho câu trả lời) — cần bổ sung để so sánh hiệu quả giữa các lựa chọn (có/không rerank, có/không domain-aware chunking).
- **Retrieval thuần dense** — có thể bổ sung hybrid search (BM25 + dense) để bắt tốt hơn các trường hợp cần khớp chính xác số Điều/Khoản/tên văn bản.
- Chưa đóng gói thành API/demo độc lập ngoài notebook Kaggle.
