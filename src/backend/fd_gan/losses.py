import torch

def classification_loss(predictions, labels):
    criterion = torch.nn.BCELoss()
    return criterion(predictions, labels)

def forgery_similarity_loss(predicted_masks, ground_truth_masks):
    return torch.mean(torch.abs(predicted_masks - ground_truth_masks))

def diversity_loss(mask_filters):
    loss = 0
    for i in range(len(mask_filters)):
        for j in range(i+1, len(mask_filters)):
            loss += torch.cosine_similarity(mask_filters[i].weight.flatten(),
                                            mask_filters[j].weight.flatten(), dim=0)
    return loss
