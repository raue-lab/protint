import os
import numpy as np
import pandas as pd
import torch, shutil, mlflow
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path

# modified from mober's script

def get_class_weights(class_series, balanced_sources):
    # classes as NumPy array (ordered)
    classes = np.array(sorted(class_series.unique()))

    if balanced_sources:
        # y must have the same label dtype as `classes`
        y = class_series.astype(str).to_numpy()
        classes = classes.astype(str)
        weights = compute_class_weight(class_weight="balanced",
                                       classes=classes,
                                       y=y)
    else:
        weights = np.ones(classes.shape[0], dtype=float)

    return weights

def create_dataloaders_from_adata(
    adata, batch_size, val_set_size, random_seed, use_sparse_mat=False, detected_layer="detected"
):
    assert 0 <= val_set_size < 1.0
    assert detected_layer in adata.layers, f"Missing layer adata.layers['{detected_layer}']"

    samples = adata.obs.index.values
    splt = int(adata.shape[0] * (1 - val_set_size))

    np.random.seed(random_seed)
    sample_inds = np.arange(len(samples))
    np.random.shuffle(sample_inds)
    np.random.seed()  # reset seed

    tr_samples = samples[sample_inds[:splt]]
    val_samples = samples[sample_inds[splt:]]

    label_encode = pd.get_dummies(sorted(adata.obs.data_source.unique()))
    label = pd.get_dummies(adata.obs.data_source)

    if use_sparse_mat:
        raise NotImplementedError("See sparse-safe version below (recommended if adata.X is sparse).")

    # --- Dense path ---
    # Avoid mutating adata.X globally; use local slices.
    X_tr = adata[tr_samples, :].X
    M_tr = adata[tr_samples, :].layers[detected_layer]

    X_val = adata[val_samples, :].X if len(val_samples) > 0 else None
    M_val = adata[val_samples, :].layers[detected_layer] if len(val_samples) > 0 else None

    # Convert matrices to dense arrays if needed
    if hasattr(X_tr, "toarray"):
        X_tr = X_tr.toarray()
    if hasattr(M_tr, "toarray"):
        M_tr = M_tr.toarray()

    if X_val is not None and hasattr(X_val, "toarray"):
        X_val = X_val.toarray()
    if M_val is not None and hasattr(M_val, "toarray"):
        M_val = M_val.toarray()

    # Ensure correct dtypes
    X_tr = np.asarray(X_tr, dtype=np.float32)
    M_tr = np.asarray(M_tr, dtype=np.bool_)  # keep bool, cast in loss or here

    train_data = TensorDataset(
        torch.from_numpy(X_tr),
        torch.from_numpy(M_tr),  # bool mask
        torch.tensor(label.loc[tr_samples, :].values, dtype=torch.float32),
    )

    if len(val_samples) > 0:
        X_val = np.asarray(X_val, dtype=np.float32)
        M_val = np.asarray(M_val, dtype=np.bool_)

        val_data = TensorDataset(
            torch.from_numpy(X_val),
            torch.from_numpy(M_val),
            torch.tensor(label.loc[val_samples, :].values, dtype=torch.float32),
        )
    else:
        val_data = None

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True) if val_data is not None else None

    return train_loader, val_loader, label_encode


def create_temp_dirs(tmp_dir):
    Path(os.path.join(tmp_dir, "models")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(tmp_dir, "metrics")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(tmp_dir, "projection")).mkdir(parents=True, exist_ok=True)


def remove_temp_dirs(tmp_dir):
    shutil.rmtree(tmp_dir)
    
    


class log_obj:
    def __init__(self, use_mlflow, run_dir):
        self.use_mlflow = use_mlflow
        self.run_dir = run_dir
        self.fhands = {}
    
    def log_params(self,args):
        if self.use_mlflow: mlflow.log_params(vars(args))
        dfparams = pd.DataFrame(data=vars(args),index=['value']).transpose()
        dfparams.to_csv(os.path.join(self.run_dir, 'models', 'params.csv'))
        
    def log_metric(self,name,value,epoch):
        if self.use_mlflow: mlflow.log_metric(name, value, step=epoch)
        else:
            if name not in self.fhands.keys():
                fhand = open(os.path.join(self.run_dir,'metrics',name),'w',buffering=1)
                fhand.write('epoch\tvalue\n')
                self.fhands[name] = fhand
                
            self.fhands[name].write(f'{epoch}\t{value}\n')
    
    def end_log(self):
        if self.use_mlflow:
            mlflow.log_artifacts(self.run_dir)
            mlflow.end_run()
            remove_temp_dirs(self.run_dir)
        else: 
            for fhand in self.fhands.values(): fhand.close()