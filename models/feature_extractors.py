import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
from copy import deepcopy

#Define the custom feature extractor (for dictionary input space)
class SingleTokenFeatureExtractor(BaseFeaturesExtractor):
    """
    This is a feature extractor for the the single token enviroments 
    This breakes up the observations and passes them through different modules
    Timeseries observations will pass through an LSTM layer
    Discrete observations will pass through an MLP
    These are the combined in an MLP.
    """
    def __init__(self, observation_space: spaces.Dict, **kwargs):
        """
        Valid Kwargs:
        (int) features_dim: The size of the features dimension. This is the output of the entire feature extractor
        (bool) compile_flag: A flag that specifies if you whish to compile the model
        (dict) timeseries_obs: The dictionary specifying the timeseries observation space
        (dict) discrete_obs: The dictionary specifying the discrete obesrvation space
        (list) disc_layers: An array of the discrete network layers
        (list) combiner_layers: An array of the combiner layers
        (int) lstm_hidden_size: The size of the LSTM hidden state
        """

        #Get the feature dimension
        features_dim = kwargs.get("features_dim", 10)

        #Initialize the base class
        super(SingleTokenFeatureExtractor, self).__init__(observation_space,features_dim)

        #Get the compile flag
        compile_flag = kwargs.get("compile_flag", False)

        #Get the continious observation space keys
        self.timeseries_keys = kwargs.get("timeseries_obs", {}).keys()

        #Get the discrete observation space keys
        self.disc_keys = kwargs.get("discrete_obs", {}).keys()

        #Get the disc net layers
        self.disc_layers = deepcopy(kwargs.get("disc_layers", []))

        #Get the combiner hidden layers
        self.combiner_layers = deepcopy(kwargs.get("combiner_layers", []))

        #Get the LSTM hidden size
        lstm_hidden_size = kwargs.get("lstm_hidden_size", 0)

        #Assert that the network has some parameters
        assert (len(self.disc_keys) != None) and (len(self.timeseries_keys) != None), "You must provide at least on observation key"
        
        #Check to see if there are any discrete keys
        if len(self.disc_keys) != 0:

            #Add input layer size to layer config
            self.disc_layers.insert(0,len(self.disc_keys)*2)

            #create network
            self.disc_net = nn.Sequential()

            #Add a flatening layer (discrete observations are by defaule dim=2)
            self.disc_net.add_module("flatten", nn.Flatten())

            #Itterate through discrete network layers
            for i in range(1,len(self.disc_layers)):

                #Add a linear and relu for each layer in the config
                self.disc_net.add_module(f"hidden_{i}", nn.Linear(self.disc_layers[i-1], self.disc_layers[i]))
                self.disc_net.add_module(f"relu_{i}", nn.ReLU())
        else:

            #If there are no discrete parameters, set to null values.
            self.disc_layers = [0]
            self.disc_keys = {}
            self.disc_net = lambda x : x
        
        #check to see if ther are any timeseries keys
        if len(self.timeseries_keys) != 0:

            #Initialise a model dictionary for each LSTM
            self.lstm_dict = nn.ModuleDict({})

            #Itterte through timeseries keys and create an LSTM for each
            for key in self.timeseries_keys:
                self.lstm_dict[key] = nn.LSTM(1, lstm_hidden_size, batch_first=True)
        else:

            #If there are not timeseries keys, revert to null values.
            self.lstm_hidden_size = [0]
            self.lstm_keys = {}
            self.lstm_dict={}

        #Set the size of the combiner input to the size of the LSTM + discrete network outputs
        self.combiner_layers.insert(0,lstm_hidden_size * len(self.timeseries_keys) + self.disc_layers[-1])

        #Set the final layer of the combiner to the features dim
        self.combiner_layers.append(features_dim)

        #Create the combiner network
        self.combiner_net = nn.Sequential()
        for i in range(1,len(self.combiner_layers)):

            #Itterate through the layers config, adding them to the model
            self.combiner_net.add_module(f"hidden_{i}", nn.Linear(self.combiner_layers[i-1], self.combiner_layers[i]))
            self.combiner_net.add_module(f"relu_{i}", nn.ReLU())
        

        ### COMPILE FOR BETTER PERFORMANCE ###
        ### Note that this causes errors when run in the an IDE ###
        if compile_flag:
            self.cont_net = torch.compile(self.cont_net)
            self.disc_net = torch.compile(self.disc_net)
            self.combiner_net = torch.compile(self.combiner_net)
            self.lstm = torch.compile(self.lstm)

    def forward(self, observations):

        #Initialize a discrete observation tensor
        disc_obs = [torch.tensor([])]

        #Itterate through each discrete obervation
        for key in self.disc_keys:

            #append to discrete observations
            disc_obs.append(observations[key])
        
        #concatenate all observations
        disc_obs = torch.cat(disc_obs, dim = -1)

        #Initialize an empty hidden state tensor
        hidden_states = [torch.tensor([])]

        #Itterate through all timeseries keys
        for key, lstm in self.lstm_dict.items():

            #unsqueeze obervations so they have dim=3 (for the LSTM)
            obs = observations[key].unsqueeze(-1)

            #extract the hidden states
            _, (hn, _) = lstm(obs)

            #append the lats hidden state to the arrat
            hidden_states.append(hn[-1])  

        #concatenate all the hidden LSTM states
        lstm_out = torch.cat(hidden_states, dim=-1)  

        #calculate the output of the discrete network
        disc_obs_out = self.disc_net(disc_obs)

        #Extract non empty tensors
        valid_tensors = [t for t in [lstm_out, disc_obs_out] if t.numel() > 0]

        #concatenate outputs for the different networks
        Y = torch.cat(valid_tensors, dim=-1) if valid_tensors else torch.tensor([])

        #Pass through the combiner network
        Z = self.combiner_net(Y) if Y.numel() > 0 else torch.tensor([])

        #return the network output
        return Z
    


class PairsFeatureExtractor(BaseFeaturesExtractor):
    """
    This is a feature extractor for the the single token enviroments 
    This breakes up the observations and passes them through different modules
    Timeseries observations will pass through an LSTM layer
    Discrete observations will pass through an MLP
    Indicator observations will pass through an MLP
    These are the combined in an MLP.
    """
    def __init__(self, observation_space: spaces.Dict, **kwargs):
        """
        Valid Kwargs:
        (int) features_dim: The size of the features dimension. This is the output of the entire feature extractor
        (bool) compile_flag: A flag that specifies if you whish to compile the model
        (dict) timeseries_obs: The dictionary specifying the timeseries observation space
        (dict) discrete_obs: The dictionary specifying the discrete obesrvation space
        (disc) indicator_obs: The dictionary defining the indicator observation space
        (list) disc_layers: An array of the discrete network layers
        (list) indicator_layers: An array of the indicator network layers
        (list) combiner_layers: An array of the combiner layers
        (int) lstm_hidden_size: The size of the LSTM hidden state
        """

        #Get the feature dimension
        features_dim = kwargs.get("features_dim", 10)

        #Initialize the base class
        super(PairsFeatureExtractor, self).__init__(observation_space,features_dim)

        #Get the compile flag
        compile_flag = kwargs.get("compile_flag", False)

        #Get the continious observation space keys
        self.timeseries_keys = kwargs.get("timeseries_obs", {}).keys()

        #get the indicator observations
        self.indicator_keys = deepcopy(kwargs.get("indicator_obs", {})).keys()

        #Get the discrete observation space keys
        self.disc_keys = kwargs.get("discrete_obs", {}).keys()

        #get the indicator network layers
        self.indicator_layers = deepcopy(kwargs.get("indicator_layers", [0]))

        #Get the LSTM hidden size
        lstm_hidden_size = kwargs.get("lstm_hidden_size", 0)

        #Get the disc net layers
        self.disc_layers = deepcopy(kwargs.get("disc_layers", [2,2]))

        #Get the combiner hidden layers
        self.combiner_layers = deepcopy(kwargs.get("combiner_layers", [10,10]))

        self.token_pair = kwargs.get("token_pair", None)

        #Assert that the network has some parameters
        assert (len (self.indicator_keys) != None) and (len(self.disc_keys) != None) and (len(self.timeseries_keys) != None), "You must provide at least on observation key"

        assert not self.token_pair is None, "You must provide the token pair to the feature extractor"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #create empty lstm keys
        self.lstm_keys = []

        #For itterate through each timeseries key
        for key in self.timeseries_keys:

            #check to see if key is z_score
            if key != 'z_score':

                #If not, create an LSTM key for each tokes
                self.lstm_keys.append(self.token_pair[0] + '_' + key)
                self.lstm_keys.append(self.token_pair[1] + '_' + key)
            else:

                #create one z_score key for both tokens
                self.lstm_keys.append('z_score')

        #If adfuller is in the indicator keys, create a key for each token
        if 'adfuller' in self.indicator_keys:
            self.indicator_keys = list(self.indicator_keys)
            self.indicator_keys.remove('adfuller')
            self.indicator_keys.append(self.token_pair[0] + '_adfuller')
            self.indicator_keys.append(self.token_pair[1]+ '_adfuller')

        #check to see if theres indicator keys
        if len(self.indicator_keys) != 0:

            #Add the indicator net input size to the config
            self.indicator_layers.insert(0,len(self.indicator_keys))

            #create the indicator network
            self.indicator_net = nn.Sequential()

            #Populate indicator network with the specified layers
            for i in range(1,len(self.indicator_layers)):
                self.indicator_net.add_module(f"layer_{i}", nn.Linear(self.indicator_layers[i-1],self.indicator_layers[i]))
                self.indicator_net.add_module(f"relu_{i}", nn.ReLU())
        else:

            #If there is no indicator keys, create a null layer
            self.indicator_layers = [0]
            self.indicator_keys = {}
            self.indicator_net = lambda x : x

        #check to see if theres discrete keys
        if len(self.disc_keys) != 0:

            #Add the discrete net input size to the config
            self.disc_layers.insert(0,len(self.disc_keys)*2)

            #create discrete network and add a flatten layer
            self.disc_net = nn.Sequential(nn.Flatten())

            #Populate discrete network with the specified layers
            for i in range(1,len(self.disc_layers)):
                self.disc_net.add_module(f"hidden_{i}", nn.Linear(self.disc_layers[i-1], self.disc_layers[i]))
                self.disc_net.add_module(f"relu_{i}", nn.ReLU())
        else:

            #If there are no discrete keys, create a null layer
            self.disc_layers = [0]
            self.disc_keys = {}
            self.disc_net = lambda x : x
        
        #check for timeseries keys
        if len(self.lstm_keys) != 0:

            #create a lstm dictionary
            self.lstm_dict = nn.ModuleDict({})

            #populate dictionary with LSTM models for each timeseries key
            for key in self.lstm_keys:
                self.lstm_dict[key] = nn.LSTM(1, lstm_hidden_size, batch_first=True)
        else:

            #If there are no lstm keys, create a null model
            self.lstm_hidden_size = [0]
            self.lstm_keys = {}
            self.lstm_dict={}

        #Add input layer to combiner model config
        self.combiner_layers.insert(0,lstm_hidden_size * len(self.lstm_keys) + self.disc_layers[-1] + self.indicator_layers[-1])

        #Add output layer to conbiner config
        self.combiner_layers.append(features_dim)

        #create combiner network
        self.combiner_net = nn.Sequential()

        #Itterare through combiner config, adding layers
        for i in range(1,len(self.combiner_layers)):
            self.combiner_net.add_module(f"hidden_{i}", nn.Linear(self.combiner_layers[i-1], self.combiner_layers[i]))
            self.combiner_net.add_module(f"relu_{i}", nn.ReLU())

        ### COMPILE FOR BETTER PERFORMANCE ###
        ### Note that this can cause errors when run in the an IDE ###
        if compile_flag:
            self.cont_net = torch.compile(self.cont_net)
            self.disc_net = torch.compile(self.disc_net)
            self.combiner_net = torch.compile(self.combiner_net)
            self.lstm = torch.compile(self.lstm)

    def forward(self, observations):

        #Initialize discrete obervation tensor
        disc_obs = [torch.tensor([]).to(self.device)]

        #itterate through all discrete observations and concatenate them
        for key in self.disc_keys:
            disc_obs.append(observations[key])
        disc_obs = torch.cat(disc_obs, dim = -1)

        #initialise indicator observation tensor
        indicator_obs = [torch.tensor([]).to(self.device)]

        #itterate through all continious observations and concatenate them
        for key in self.indicator_keys:
            indicator_obs.append(observations[key])
        indicator_obs = torch.cat(indicator_obs, dim = -1)

        #create empty hidden state tensor
        hidden_states = [torch.tensor([]).to(self.device)]

        #Apply LSTM to each timeseries observation and extract the final hidden state
        for key, lstm in self.lstm_dict.items():
            obs = observations[key].unsqueeze(-1)
            _, (hn, _) = lstm(obs)
            hidden_states.append(hn[-1])  

        #concatenate all hidden states
        lstm_out = torch.cat(hidden_states, dim=-1)  

        #propagate through the discrete and indicator networks
        disc_obs_out = self.disc_net(disc_obs)
        indicator_obs_out = self.indicator_net(indicator_obs)

        #concatenate only populated tensors
        valid_tensors = [t for t in [lstm_out, disc_obs_out, indicator_obs_out] if t.numel() > 0]
        Y = torch.cat(valid_tensors, dim=-1) if valid_tensors else torch.tensor([])

        #Pass through the combiner network
        Z = self.combiner_net(Y) if Y.numel() > 0 else torch.tensor([])

        #return output
        return Z
