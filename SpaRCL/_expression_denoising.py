# -*- coding: utf-8 -*-

from typing import Optional
from anndata import AnnData

import numpy as np
from scipy.sparse import issparse, csr_matrix


def expression_denoising(
    adata: AnnData,
    relation_key: Optional[str] = None,
) -> AnnData:
    '''
    Denoise the gene expression matrix using gene relation matrix.
    
    Parameters
    ----------
    adata
        Annotated data matrix.
    relation_key
        If not specified, it looks `.uns['relation']` for relational contrastive learning settings
        (default storage place for :func:`~SpaRCL.run_RCL`).
        If specified, it looks `.uns[relation_key]` for relational contrastive learning settings.
    
    Returns
    -------
    Returns ``adata`` of used genes with the following fields updated.
    
    .X
        The denoised gene expression matrix.
    .layers['original']
        The original gene expression matrix.
    '''
    
    if relation_key is None:
        relation_key = 'relation'
    
    if relation_key not in adata.uns:
        raise ValueError(f'Did not find .uns["{relation_key}"].')
    
    
    adata_denoised = adata[:, adata.uns[relation_key]['gene_names']].copy()
    
    adata_denoised.layers['original'] = adata_denoised.X.copy()
    
    adata_denoised.X = csr_matrix(np.matmul(
        adata_denoised.X.toarray() if issparse(adata_denoised.X) else adata_denoised.X,
        adata_denoised.uns[relation_key]['gene_relation'].toarray(),
    ))
    
    return adata_denoised



















