# AI Use Disclosure - HW6

## 1. How I used AI

- Used Gemini to help me understand the concepts behind the assignment:
  - Why self-supervised learning helps when only 500 labels are available
  - How rotation prediction works as a pretext task
  - How SimCLR and NT-Xent loss pull augmented views together and push other images apart
  - What linear probing means (frozen encoder, train only a linear layer)
  - How the rotation labels are generated for each unlabeled image
  - How the two augmented views are built for SimCLR
  - How precision at 5 is computed for the nearest neighbor part


## 2. Something the AI got wrong

- Gemini first told me the linear probe also updates the encoder weights.
- That is wrong for this assignment. The encoder is frozen and only the final linear layer is trained.
- I caught it by checking the notebook, where the probe is trained on stored embeddings, so the encoder cannot change.

## 3. How I checked everything

- I compared every explanation against my own executed notebook outputs.
- I read the loss, rotation accuracy, and test accuracy for each part myself.
- I compared the accuracy and precision at 5 across all three models in my own table.
- I looked at the nearest neighbor images myself to see which matched the query class.
