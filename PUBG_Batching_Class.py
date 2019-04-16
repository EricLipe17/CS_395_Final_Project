import pandas as pd


# Create a class that will do the batching for the algorithm
class PUBG_Data_Reader():
    pd.set_option('display.max_columns', None)

    # Dataset is a mandatory arugment, while the batch_size is optional
    # If you don't input batch_size, it will automatically take the value: None
    def __init__(self, dataset, batch_size=None):

        train = pd.read_csv(dataset)
        
        # Two variables that take the values of the inputs and the targets. Inputs are floats, targets are floats
        self.inputs, self.targets = pd.DataFrame(train.iloc[:, 2:68]), pd.DataFrame(train.iloc[:, -1])
        self.inputs_shape = self.inputs.shape
        self.targets_shape = self.targets.shape
        
        # Counts the batch number, given the size you feed it later
        # If the batch size is None, we are either validating or testing, so we want to take the data in a single batch
        if batch_size is None:
            self.batch_size = self.inputs.shape[0]
        else:
            self.batch_size = batch_size
        self.curr_batch = 0
        self.batch_count = self.inputs.shape[0] // self.batch_size

    # Returns the shape of the inputs and targets
    def shape_of(self):
        return self.inputs_shape, self.targets_shape

    # Returns the first 5 rows of the dataset
    def inputs_head(self):
        return self.inputs.head()
    
    # A method which loads the next batch
    def __next__(self):
        if self.curr_batch >= self.batch_count:
            self.curr_batch = 0
            raise StopIteration()
            
        # You slice the dataset in batches and then the "next" function loads them one after the other
        batch_slice = slice(self.curr_batch * self.batch_size, (self.curr_batch + 1) * self.batch_size)
        inputs_batch = self.inputs[batch_slice]
        targets_batch = self.targets[batch_slice]
        self.curr_batch += 1
        
        # The function will return the inputs batch and the targets batch
        return inputs_batch, targets_batch

    # A method needed for iterating over the batches, as we will put them in a loop
    # This tells Python that the class we're defining is iterable, i.e. that we can use it like:
    # for input, output in data: 
    # do things
    # An iterator in Python is a class with a method __next__ that defines exactly how to iterate through its objects
    def __iter__(self):
        return self

