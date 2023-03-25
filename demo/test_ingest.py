from lsst.daf.butler import Butler, DatasetType, FileDataset, StorageClass, DatasetRef, Formatter

# Set up the Butler
butler = Butler("/repo/apv", run="pretrained_models")

# Print all known storage classes
#print(butler.registry.storageClasses)
#exit()

dataset_type = 'pretrainedModel'
data_id = {'instrument': 'LSSTCam-imSim'}

# Register the dataset type
dataset_type = DatasetType(dataset_type,
                           dimensions=['instrument'],
                           storageClass='StructuredDataDict',
                           universe=butler.registry.dimensions)
butler.registry.registerDatasetType(dataset_type)

# Load a binary file into the memory as a big blob of bytes
def read_data(filename):
    with open(filename, "rb") as f:
        return f.read()

def read_pytorch_checkpoint(filename):
    import torch
    return torch.load(filename)


# Create a Formatter class, which reads and writes binary data
class BinaryFormatter(Formatter):
    extension = ".tar"

    def read(self, component=None):
        return read_data(self.fileDescriptor.location.path)

    def write(self, inMemoryDataset):
        with open(self.fileDescriptor.location.path, "wb") as f:
            f.write(inMemoryDataset)

class PytorchCheckpointFormatter(Formatter):
    extension = ".pth.tar"

    def read(self, fileDescriptor):
        import torch
        return torch.load(fileDescriptor.location.path)

    def write(self, inMemoryDataset):
        import torch
        torch.save(inMemoryDataset, self.fileDescriptor.location.path)

# Create a FileDataset instance from the file
file_path = 'file:///home/nima/lsst-repos/meas_transiNet/model_packages/dummy/checkpoint.pth.tar'
file_dataset = FileDataset(path=file_path,
                           refs=DatasetRef(dataset_type, data_id),
                           formatter=BinaryFormatter)

# Ingest the file into the butler
butler.ingest(file_dataset,
              transfer='copy')

# butler.put(obj = read_pytorch_checkpoint(file_path),
#            datasetRefOrType=dataset_type,
#            dataId=data_id)
