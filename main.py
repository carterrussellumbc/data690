from model import ImageClassifier

# lr = 0.00005 | opt = adam
# accuracy: 0.7521 - loss: 0.5066 - val_accuracy: 0.7988 - val_loss: 0.4454 | nothing
# 0.7573 no flip
# 0.7377 flip
# 0.5646 rdm brt [-1.0, 1.0]
# 0.5763 rdm brt [-0.2, 0.2]

# lr = 0.0001 | opt = adam | better optimizer
# accuracy: 0.7579 - loss: 0.5017 - val_accuracy: 0.8050 - val_loss: 0.4329 | nothing
# accuracy: 0.8451 - loss: 0.3572 - val_accuracy: 0.8487 - val_loss: 0.3541 | nothing Epoch = 5
# accuracy: 0.7635 - loss: 0.4941 - val_accuracy: 0.7318 - val_loss: 0.6415 | rotation 0.2
# accuracy: 0.7495 - loss: 0.5075 - val_accuracy: 0.8041 - val_loss: 0.4371 | flip
# accuracy: 0.8385 - loss: 0.3680 - val_accuracy: 0.8461 - val_loss: 0.3567 | flip Epoch = 5

# lr = 0.0001 | opt = adamax | all around worse
# accuracy: 0.8122 - loss: 0.4155 - val_accuracy: 0.8145 - val_loss: 0.4116 | flip Epoch = 5
# accuracy: 0.8126 - loss: 0.4148 - val_accuracy: 0.8160 - val_loss: 0.4090 | nothing Epoch = 5


img_data_dir = "C:\\Users\\Carter\\Downloads\\histopathologic-cancer-detection\\data\\jpg"
classifier = ImageClassifier()
classifier.train_model(0.1, 96024, img_data_dir, 2, 1, 'cancer_classifier.keras')
