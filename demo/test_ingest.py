from lsst.daf.butler import Butler, DatasetType, FileDataset, StorageClass, DatasetRef
from lsst.meas.transiNet.modelPackages.storageAdapterButlerHybrid import BinaryFormatter, PytorchCheckpointFormatter

# Set up the Butler
butler = Butler("/repo/apv", run="pretrained_models/dummy")

# Print all known storage classes
#print(butler.registry.storageClasses)
#exit()

dataset_type_name = 'pretrainedModel'
storage_class = "StructuredDataDict"
data_id = {'instrument': 'LSSTCam-imSim'}

# # Find the first such dataset in the registry.
# dataset_ref = butler.registry.findDataset(dataset_type, data_id, collections=["pretrained_models"],)
# butler.getDirect(dataset_ref)
# exit()


# Create the dataset type
dataset_type = DatasetType(dataset_type_name,
                           dimensions=['instrument'],
                           storageClass=storage_class,
                           universe=butler.registry.dimensions)

def register_dataset_type(butler, dataset_type_name, storage_class):
    try:
        butler.registry.getDatasetType(dataset_type_name)
    except KeyError:
        butler.registry.registerDatasetType(dataset_type)
        return dataset_type

def do_ingestion(butler, dataset_type, data_id, file_path):
    # Create a FileDataset instance from the file.
    # The dataset needs to be igested into a specific collection.

    file_dataset = FileDataset(path=file_path,
                               refs=DatasetRef(dataset_type, data_id),
                               #formatter=BinaryFormatter,
                               formatter=PytorchCheckpointFormatter,
                               )

    # Ingest the file into the butler
    butler.ingest(file_dataset,
                  run="pretrained_models/dummy",
                  transfer='copy')


file_path = 'file:///home/nima/lsst-repos/meas_transiNet/model_packages/dummy/checkpoint.pth.tar'
register_dataset_type(butler, dataset_type_name, storage_class)
do_ingestion(butler, dataset_type, data_id, file_path)
