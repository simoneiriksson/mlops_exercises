import torch
def test_data():
    dataset_train = torch.load("data/processed/train_images.pt")
    print(len(dataset_train))
    assert len(dataset_train) == 50000 
    dataset_test = torch.load("data/processed/test_images.pt")
    #print(len(dataset_test))
    assert len(dataset_test) == 5000 
    print(dataset_train.shape)
    assert dataset_train[0].shape == torch.Size([1, 28, 28]) 

    dataset_train_target = torch.load("data/processed/train_labels.pt")    
    assert dataset_train_target.unique().shape[0] == 10

    dataset_test_target = torch.load("data/processed/test_labels.pt")    
    assert dataset_test_target.unique().shape[0] == 10
