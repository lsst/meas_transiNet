# Pseudocode sketch of the classifier Task


class ClassifierConfig:
    # various settings
    classifier = RbTransinetInterface


class ClassifierConnections:
    Diffim input
    diaSource input
    Template input
    science input
    pretrainedModel input

    diaSourceScored output


class ClassifierTask:
    def __init__:
        # set things up
        self.makeSubTask("classifier")

    def run(self, inputs):
        self.classifier.init(pretrainedModel)

        for src in diaSources:
            # Extract cutouts around each diaSource. These should be numpy arrays in units of nJy
            diffimCutout = ...
            scienceCutout = ...
            templateCutout = ...

            inputs = [[diffimCutout, scienceCutout, templateCutout], ]
            scores = self.classifier.infer(inputs)

            update diaSource with score

        results = updated diaSource table

        return results
