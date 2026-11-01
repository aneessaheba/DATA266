"""HW4.a part 1: fine tune BERT on the Twitter slice, evaluate, save, reload, custom inference."""
import os, time
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from transformers import BertForSequenceClassification, BertTokenizer
from common import *

seed_all()
tr, va, te, test_texts, names, info = load_data()
g = torch.Generator().manual_seed(SEED)
train_dl = DataLoader(tr, batch_size=BATCH, shuffle=True, generator=g)
val_dl, test_dl = DataLoader(va, batch_size=128), DataLoader(te, batch_size=128)

model = build_model(len(names))
opt, _ = make_opt(model)
scaler = make_scaler(False)
epochs = []
reset_peak()
t0 = time.time()
for ep in range(1, EPOCHS + 1):
    t = time.time()
    tl, _ = train_epoch(model, train_dl, opt, scaler)
    vl, vacc, _, _ = evaluate(model, val_dl)
    epochs.append(dict(epoch=ep, train_loss=tl, val_loss=vl, val_acc=vacc, epoch_s=time.time() - t))
    print(epochs[-1], flush=True)
train_s, peak = time.time() - t0, peak_gb()

# save and reload, then everything below uses the reloaded model
path = os.path.join(ROOT, "models", f"bert_twitter_{CARD}")
model.save_pretrained(path)
BertTokenizer.from_pretrained(BERT).save_pretrained(path)
model = BertForSequenceClassification.from_pretrained(path).to(DEV)
tok = BertTokenizer.from_pretrained(path)

_, test_acc, preds, ys = evaluate(model, test_dl)
p, r, f1, n = precision_recall_fscore_support(ys, preds, zero_division=0)
worst = int(f1.argmin())
wrong = [dict(text=test_texts[i], true=names[ys[i]], pred=names[preds[i]])
         for i in range(len(ys)) if ys[i] == worst and preds[i] != worst][:5]

custom = ["I love how fast this new phone charges!", "Traffic again. Worst commute ever.",
          "The meeting is moved to 3pm on Thursday."]
enc = tok(custom, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(DEV)
with torch.no_grad():
    guess = model(**enc).logits.argmax(-1).tolist()

save("hw4a", dict(data=info, settings=dict(model=BERT, max_len=MAX_LEN, batch=BATCH, lr=LR, epochs=EPOCHS, seed=SEED),
                  epochs=epochs, train_s=train_s, peak_gb=peak, test_acc=test_acc,
                  confusion=confusion_matrix(ys, preds).tolist(),
                  per_class=[dict(cls=names[i], precision=float(p[i]), recall=float(r[i]), f1=float(f1[i]), n=int(n[i]))
                             for i in range(len(names))],
                  worst_class=names[worst], misclassified=wrong,
                  custom=[dict(text=c, pred=names[k]) for c, k in zip(custom, guess)]))
print("test acc", test_acc, "peak GB", peak, "train s", train_s)
