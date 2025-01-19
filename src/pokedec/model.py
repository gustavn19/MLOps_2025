import timm

def get_model(num_classes: int):
    '''
    Creates a ResNet50 model pre-trained on ImageNet with a custom number of output classes.
    
    Args:
        num_classes (int): The number of output classes for the classifier.
    
    Returns:
        model: A ResNet50 model with the specified number of output classes.
    '''
    # Load pretrained model
    model = timm.create_model('resnet50d.ra4_e3600_r224_in1k', pretrained=True)

    # Set new number of classes
    model.reset_classifier(num_classes=num_classes)

    # Freeze base layers
    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True

    return model
