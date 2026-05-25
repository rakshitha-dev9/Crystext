# CrysText — How To Run Training 

Hi! Follow these steps exactly. Do NOT skip any step.

---

## PART 1 — One Time Setup (Do this once)

### Step 1 — Create a HuggingFace Account
- Go to https://huggingface.co
- Sign up for a free account

### Step 2 — Create a HuggingFace Token
- Go to https://huggingface.co/settings/tokens
- Click "New Token"
- Name it anything (e.g. "kaggle-token")
- Role → select "Write"
- Click Generate
- **Copy the token — it starts with hf_ — you won't see it again**

### Step 3 — Open the Notebook on Kaggle
- Go to https://kaggle.com
- Sign in or create a free account
- Upload the notebook file `crystext_training.ipynb`

### Step 4 — Enable GPU on Kaggle
- On the right side panel → click "Session Options"
- Accelerator → select **GPU T4 x1**
- Internet → make sure it is **ON**

### Step 5 — Add Your HuggingFace Token to Kaggle Secrets
- On the right side panel → click "Add-ons" → "Secrets"
- Click "Add a new secret"
- Name: `HF_TOKEN` (must be exactly this, capital letters)
- Value: paste your token from Step 2
- Click Save

---

## PART 2 — Before Running, Edit These Two Things

### Edit 1 — Change the session rows
Find this line in the data loading cell:
```python
train_df = train_df.iloc[0:10000].reset_index(drop=True)
```

Change it based on which session you are running:
- Session 4 → `iloc[10000:20000]`
- Session 5 → `iloc[20000:27136]`

**Session 3 (rows 0-10000) is already done. Do not repeat it.**

### Edit 2 — Change the HuggingFace repo name to yours
Find this in Cell 11:
```python
model.push_to_hub("rakshitha9/crystext-mistral-10k", ...)
tokenizer.push_to_hub("rakshitha9/crystext-mistral-10k", ...)
```

Change `rakshitha9` to **your HuggingFace username**:
```python
model.push_to_hub("YOURUSERNAME/crystext-mistral-10k", ...)
tokenizer.push_to_hub("YOURUSERNAME/crystext-mistral-10k", ...)
```

---

## PART 3 — Run the Notebook

- Click "Run All" at the top
- Training takes approximately **9 hours** on Kaggle T4
- Do NOT close the browser tab while training
- When done you will see:
```
✅ Training complete!
⏱️ Time: XX minutes
📉 Loss: 0.XXXX
```
- After that Cell 11 pushes the model to YOUR HuggingFace account automatically

---

## PART 4 — After Training

- Go to https://huggingface.co/YOURUSERNAME/crystext-mistral-10k
- You should see your model there
- Share the model link with the team so we can update app.py

---

## Common Errors

| Error | Fix |
|---|---|
| "No GPU available" | Enable T4 GPU in Kaggle session options |
| "Secret HF_TOKEN not found" | Add your token in Kaggle Add-ons → Secrets |
| "Repository not found" | Make sure your HuggingFace username is correct in Cell 11 |
| "CUDA out of memory" | Restart the notebook and run again — don't run cells twice |
| Training stops midway | Kaggle sessions expire after 12 hours — start from the session that was pending |

---

## Questions?
Contact Rakshitha or check the main README.md in the GitHub repo.
