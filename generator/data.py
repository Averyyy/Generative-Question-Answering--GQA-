from torch.utils.data import Dataset


class GenerativeQADataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return self.dataset["validation"].num_rows  # specify the split

    def __getitem__(self, idx):
        row = self.dataset["validation"][idx]  # specify the split
        question = row["question"]
        best_answer = row["best_answer"]
        correct_answers = row["correct_answers"]
        incorrect_answers = row["incorrect_answers"]
        source_text = row["source_text"]
        return question, best_answer, correct_answers, incorrect_answers, source_text
