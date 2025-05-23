# Source from:
# [1] https://github.com/saprmarks/geometry-of-truth/blob/main/utils.py
# [2] https://github.com/saprmarks/geometry-of-truth/blob/main/visualization_utils.py

import os
from glob import glob

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch as t

ROOT = os.path.dirname(os.path.abspath(__file__))
ACTS_BATCH_SIZE = 25


def get_pcs(X, k=2, offset=0):
    """
    Performs Principal Component Analysis (PCA) on the n x d data matrix X.
    Returns the k principal components, the corresponding eigenvalues and the projected data.
    """

    # Subtract the mean to center the data
    X = X - t.mean(X, dim=0)

    # Compute the covariance matrix
    cov_mat = t.mm(X.t(), X) / (X.size(0) - 1)

    # Perform eigen decomposition
    eigenvalues, eigenvectors = t.linalg.eigh(cov_mat)

    # Since the eigenvalues and vectors are not necessarily sorted, we do that now
    sorted_indices = t.argsort(eigenvalues, descending=True)
    eigenvectors = eigenvectors[:, sorted_indices]

    # Select the pcs
    eigenvectors = eigenvectors[:, offset : offset + k]

    return eigenvectors


def collect_acts(
    dataset_name, model, layer, noperiod=False, center=True, scale=False, device="cpu"
):
    """
    Collects activations from a dataset of statements, returns as a tensor of shape [n_activations, activation_dimension].
    """
    directory = os.path.join(ROOT, "acts", model)
    if noperiod:
        directory = os.path.join(directory, "noperiod")
    directory = os.path.join(directory, dataset_name)
    activation_files = glob(os.path.join(directory, f"layer_{layer}_*.pt"))
    if len(activation_files) == 0:
        raise ValueError(f"Dataset {dataset_name} not found.")
    acts = [
        t.load(os.path.join(directory, f"layer_{layer}_{i}.pt")).to(device)
        for i in range(0, ACTS_BATCH_SIZE * len(activation_files), ACTS_BATCH_SIZE)
    ]
    acts = t.cat(acts, dim=0).float().to(device)
    if center:
        acts = acts - t.mean(acts, dim=0)
    if scale:
        acts = acts / t.std(acts, dim=0)
    return acts


class TruthData:
    """
    A dataset consisting of factual statements, their truth values, and their representations when run through a LM.
    """

    # df is a pandas dataframe
    def __init__(self, df):
        self.df = df
        self.model = None
        self.layer = None

    def __len__(self):
        return len(self.df)

    @classmethod
    def from_datasets(
        cls,
        dataset_names,
        model,
        layer,
        noperiod=False,
        center=True,
        scale=False,
        device="cpu",
    ):
        dfs = []
        for dataset_name in dataset_names:
            df = pd.read_csv(os.path.join("datasets", f"{dataset_name}.csv"))

            # append activations to df
            acts = collect_acts(
                dataset_name,
                model,
                layer,
                noperiod=noperiod,
                center=center,
                scale=scale,
                device=device,
            ).cpu()
            try:
                df["activation"] = list(acts)
            except:
                raise ValueError(f"Issue with dataset {dataset_name}")

            dfs.append(df)

        df = pd.concat(dfs, keys=dataset_names)

        out = TruthData(df)
        out.model = model
        out.layer = layer

        return out

    # make a scatterplot of the data after dimensionality reduction
    # dimensions : number of dimensions to reduce to
    # dim_offset : how many of the top PCs to ignore
    # label : column of df to use as labels
    # plot_datasets : dataset to use for plotting (by default, use all data)
    # pca_datasets : dataset to use for PCA (by default, use all data)
    def plot(
        self,
        dimensions,
        dim_offset=0,
        plot_datasets=None,
        pca_datasets=None,
        arrows=[],
        return_df=False,
        **kwargs,
    ):
        # get pcs for the given datasets
        if pca_datasets is None:
            pca_datasets = self.df.index.levels[0].tolist()
        acts = self.df.loc[pca_datasets]["activation"].tolist()
        acts = t.stack(acts, dim=0).cuda()
        pcs = get_pcs(acts, dimensions, offset=dim_offset)

        # project data onto pcs
        if plot_datasets is None:
            plot_datasets = self.df.index.levels[0].tolist()
        df = self.df.loc[plot_datasets]
        acts = df["activation"].tolist()
        acts = t.stack(acts, dim=0).cuda()
        proj = t.mm(acts, pcs)

        # add projected data to df
        for dim in range(dimensions):
            df[f"PC{dim + 1}"] = proj[:, dim].tolist()

        # shuffle rows of df
        df = df.sample(frac=1)

        # plot using plotly
        if dimensions == 2:
            fig = px.scatter(
                df,
                x="PC1",
                y="PC2",
                hover_name="statement",
                color_continuous_scale="Bluered_r",
                **kwargs,
            )
        elif dimensions == 3:
            fig = px.scatter_3d(
                df,
                x="PC1",
                y="PC2",
                z="PC3",
                hover_name="statement",
                color_continuous_scale="Bluered_r",
                **kwargs,
            )
        else:
            raise ValueError("Dimensions must be 2 or 3")

        fig.update_yaxes(
            scaleanchor="x",
            scaleratio=1,
        )

        fig.update_layout(
            coloraxis_showscale=False,
        )

        if arrows != []:
            for i, arrow in enumerate(
                arrows
            ):  # arrow is a tensor of shape [acts.shape[1]]
                arrow = arrow.to(pcs.device)
                arrow = t.mm(arrow.unsqueeze(0), pcs)
                arrow = go.layout.Annotation(
                    x=arrow[0, 0],
                    y=arrow[0, 1],
                    xref="x",
                    yref="y",
                    axref="x",
                    ayref="y",
                    ax=0,
                    ay=0,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="#636363",
                    opacity=0.8,
                    showarrow=True,
                )
                arrows[i] = arrow

            fig.update_layout(annotations=arrows)

        if return_df:
            return fig, df
        else:
            return fig
