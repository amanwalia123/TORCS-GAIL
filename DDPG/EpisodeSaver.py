import pickle

class EpiosdeSaver(object):

    def __init__(self,file_path,fields=[]):
        self.file = file_path
        self.fields = fields

        # making dictionary for saving the data
        self.save_dict = { field:[] for field in self.fields }

    def add(self,field_values):
        for val,field in zip(field_values,self.fields):
            self.save_dict[field].append(val)

    def save_file(self):
        pickle.dump(self.save_dict, open(self.file, 'wb'))
