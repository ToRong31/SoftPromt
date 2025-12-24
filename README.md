# T5 Prompt Tuning (PEFT) cho Phân Loại Đa Nhãn

## 📖 Giới thiệu

Project này sử dụng mô hình **T5-base** với kỹ thuật **Prompt Tuning (PEFT - Parameter Efficient Fine-Tuning)** để giải quyết bài toán phân loại đa nhãn (multi-label classification) dưới dạng câu hỏi **Yes/No**.

### Cách hoạt động

Thay vì fine-tune toàn bộ tham số của mô hình T5, Prompt Tuning chỉ huấn luyện một số lượng nhỏ "virtual tokens" (tokens ảo) được thêm vào đầu input. Điều này giúp:
- Giảm đáng kể số lượng tham số cần huấn luyện
- Tiết kiệm bộ nhớ và thời gian training
- Dễ dàng lưu trữ và chia sẻ mô hình (chỉ cần lưu adapter)

Mỗi mẫu dữ liệu sẽ được chuyển thành prompt dạng:

```
{comment}

Is the text above {label}?
```

Model sẽ trả lời **"yes"** hoặc **"no"** cho từng nhãn.

### Các nhãn phân loại

Project hỗ trợ 8 nhãn:
- `antagonize`: Thái độ đối kháng
- `condescending`: Coi thường
- `dismissive`: Thờ ơ, bác bỏ
- `generalisation`: Khái quát hóa
- `generalisation_unfair`: Khái quát hóa không công bằng
- `healthy`: Lành mạnh
- `hostile`: Thù địch
- `sarcastic`: Châm biếm

---

## 📁 Cấu trúc thư mục

```
SoftPromt/
│
├── README.md              # Tài liệu hướng dẫn
├── requirements.txt       # Các thư viện cần thiết
│
├── data/                  # Thư mục chứa dữ liệu
│   ├── train.csv          # Dữ liệu training
│   ├── val.csv            # Dữ liệu validation
│   └── test.csv           # Dữ liệu test
│
└── src/                   # Thư mục chứa source code
    ├── config.py          # Cấu hình model và training
    └── train.py           # Script huấn luyện chính
```

### Mô tả files

- **data/**: Chứa file CSV với cột `comment` (văn bản) và các cột nhãn (0/1)
- **src/config.py**: Định nghĩa đường dẫn dữ liệu, tham số model, và hyperparameters
- **src/train.py**: Chứa class `T5PromptLabelTrainer` và logic training

---

## ⚙️ Cài đặt

### 1. Tạo môi trường ảo (khuyến nghị)

```bash
python -m venv venv
venv\Scripts\activate    # Windows
# source venv/bin/activate  # Linux/Mac
```

### 2. Cài đặt các thư viện

```bash
pip install -r requirements.txt
```

Các thư viện chính:
- `transformers`: Framework Hugging Face cho T5
- `peft`: Thư viện Parameter-Efficient Fine-Tuning
- `datasets`: Xử lý dữ liệu
- `torch`: PyTorch backend
- `scikit-learn`: Tính toán metrics (AUC-ROC)
- `pandas`: Đọc và xử lý CSV

---

## 🚀 Chạy Training

### Huấn luyện một nhãn (single label)

Sử dụng tham số `--label` để chỉ huấn luyện cho một nhãn cụ thể:

```bash
python src/train.py --label hostile
```

Ví dụ trên sẽ chỉ train model để phân loại nhãn **hostile**.

### Huấn luyện nhiều nhãn (multiple labels)

Sử dụng tham số `--labels` (số nhiều) để huấn luyện nhiều nhãn cùng lúc:

```bash
python src/train.py --labels hostile sarcastic antagonize
```

### Huấn luyện tất cả các nhãn

Không truyền `--label` hoặc `--labels` để train tất cả 8 nhãn:

```bash
python src/train.py
```

### Tùy chỉnh hyperparameters

Bạn có thể override các tham số mặc định:

```bash
python src/train.py --label healthy --epochs 10 --lr 0.001 --train_bs 32 --num_virtual_tokens 20
```

**Các tham số có thể tùy chỉnh:**
- `--epochs`: Số epoch huấn luyện (mặc định: 5)
- `--lr`: Learning rate (mặc định: 0.005)
- `--train_bs`: Batch size cho training (mặc định: 16)
- `--num_virtual_tokens`: Số lượng virtual tokens (mặc định: 10)
- `--fp16`: Sử dụng mixed precision training
- `--no_eval_each_epoch`: Tắt evaluation sau mỗi epoch

### Ví dụ đầy đủ

```bash
# Train 2 nhãn với cấu hình tùy chỉnh
python src/train.py --labels dismissive condescending --epochs 8 --lr 0.003 --train_bs 24 --fp16
```

---

## 📊 Kết quả Training

Trong quá trình training, bạn sẽ thấy:
- Loss và learning rate sau mỗi logging step
- AUC-ROC score cho từng nhãn sau mỗi epoch
- AUC-ROC macro (trung bình) trên validation set
- Đánh giá trên test set trước và sau khi train

Model adapter (chỉ virtual tokens) sẽ được lưu tại thư mục được chỉ định trong `config.py`:
- Mặc định: `./t5_prompt_adapter_selected_labels`

---

## 💡 Lưu ý

1. **Dữ liệu**: File CSV phải có cột `comment` và các cột nhãn với giá trị 0 hoặc 1
2. **GPU**: Khuyến nghị sử dụng GPU để tăng tốc độ training
3. **Memory**: Prompt tuning tiết kiệm bộ nhớ hơn nhiều so với full fine-tuning
4. **Adapter**: Chỉ cần lưu và chia sẻ adapter (~KB) thay vì toàn bộ model (~GB)

---

## 📝 Tham khảo

- [PEFT Library](https://github.com/huggingface/peft)
- [T5 Model](https://huggingface.co/t5-base)
- [Prompt Tuning Paper](https://arxiv.org/abs/2104.08691)

