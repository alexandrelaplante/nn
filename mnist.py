from data import DataSet, LabeledData
import nielsen.mnist_loader as mnist_loader

class MNISTLoader:
    @staticmethod
    def convert(l) -> list[LabeledData]:
        data = []
        for value, label in l:
            data.append(
                LabeledData(value=value, label=label)
            )
        return data

    @classmethod
    def load(cls) -> DataSet:
        training, validation, test = mnist_loader.load_data_wrapper()

        return DataSet(
            training=cls.convert(training),
            validation=cls.convert(validation),
            test=cls.convert(test),
        )

if __name__ == '__main__':
    data = MNISTLoader.load()

    print(data.training[0].label)
    print(data.validation[0].label)
    print(data.test[0].label)