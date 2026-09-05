"""DR-Triage source package.

Modules are added milestone by milestone:
    data.py          (M1)  loading, stratified split, PyTorch Dataset
    preprocess.py    (M2)  crop / green channel / CLAHE / Ben Graham
    augment.py       (M3)  albumentations pipelines + mixup / cutmix
    model.py         (M4)  EfficientNetV2 + CBAM + stage/ordinal/referable heads
    losses.py        (M4)  focal + CORN ordinal + QWK-aware
    train.py         (M5)  two-phase training loop
    evaluate.py      (M6)  metrics, Grad-CAM, calibration, uncertainty
    quality_gate.py  (M7)  ungradable-image rejection
    chatbot.py       (M7)  FAISS retrieval + cited answers
"""
