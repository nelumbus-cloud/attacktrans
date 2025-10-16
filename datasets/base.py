# datasets/base.py


class BaseDataset:
    def __init__(self, root, name):
        self.root = root
        self.name = name
        self.verbose = True

        #hyper parameters
        self.hidden_channels = 8
        self.num_heads = 8
        self.dropout = 0.6
        self.lr = 0.005
        self.weight_decay = 5e-4
        self.epochs = 200
        self.dataset = None

        #dataset features
        self.num_nodes = None
        self.num_node_features = None
        self.num_classes = None
        self.degrees = None

        #transform different splits
        self.transform = lambda seed : T.Compose([T.NormalizeFeatures(), lambda x: split_data(x, seed) ])
        
        


    def fit_surrogate(self, surrogate):
        
        data = Pyg2Dpr(self.dataset[0])
            
        adj, features, labels = data.adj, data.features, data.labels


        #CONTRIBUTE: this is open source issue
        
        adj, features = list(map(csr_matrix, [adj, features]))
    
        idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
      
        # fit into Surrogate model
        
        surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)

        #target nodes
        
        target_nodes = select_nodes(surrogate, data.idx_test, data.labels)  

        print(target_nodes)
        
        return target_nodes
        
        #return self.check_and_resample(target_nodes, data.idx_test)


    #attack_model should have attack
    def attack(self, attack_model, target_node, n_perturbations):
                
        data = Pyg2Dpr(self.dataset)
        adj, features, labels = data.adj, data.features, data.labels
        adj, features = list(map(csr_matrix, [adj, features]))
        
        attack_model.attack(features, adj, labels, target_node, n_perturbations)
        
        return attack_model.modified_adj, attack_model.modified_features


    def get_degrees(self):
        """Helper function to calculate and store degrees."""
        if self.degrees is None:
            adj = Pyg2Dpr(self.dataset).adj
            self.degrees = adj.sum(1).A1 # .A1 to flatten
        return self.degrees

    ## this is done for case where deg<=2 during random sampling
    
    def check_and_resample(self, targets, idx_test):

        

        new_targets = []
        degrees = self.get_degrees()
        
        candidates = list(set(idx_test) - set(targets))
        
        for target in targets:
            t = target
            while degrees[t] <= 1:
                if self.verbose:
                    print(f"{target} is removed")
                new_target = np.random.choice(candidates)
                candidates.remove(new_target)
                t = new_target
                
            new_targets.append(t)

        if self.verbose:
            print("Target Nodes", targets)
            print("New targets Nodes",new_targets)
            
        return new_targets
        
            
        
            
    
    
