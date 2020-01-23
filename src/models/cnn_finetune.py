from cnn_finetune import make_model

from src.models.classifiers import Classifier


def get_cnn_finetune_model(model_name,
                           pretrained=True,
                           dropout_p=None):
    model = make_model(
        model_name,
        1,
        pretrained=pretrained,
        dropout_p=dropout_p,
        classifier_factory=Classifier
    )
    return model
