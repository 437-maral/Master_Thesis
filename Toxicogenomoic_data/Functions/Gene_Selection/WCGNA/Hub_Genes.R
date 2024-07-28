####get hub genes
library(WGCNA)
library(data.table)
library(gridExtra)
library(ggplot2)
library(dplyr)
library(tibble)
library(devtools)
library(CorLevelPlot)

gene_expr=read.csv('gene_expression.csv')
rownames(gene_expr) <- gene_expr$X
gene_expr <- gene_expr[, -1]

module_eigengenes=read.csv('Soft_Power_Threshold/modules_18.csv')
rownames(module_eigengenes) <- module_eigengenes$X
module_eigengenes<- module_eigengenes[, -1]

###colors
moduleColors = read.csv('modules_18.csv')

###
nSamples <- nrow(gene_expr)

####correlation
module.genes.corr <- cor(gene_expr, module_eigengenes, use = "p")

######p-value
module.membership.measure.pvals <- corPvalueStudent(module.genes.corr , nSamples)


unsig_modules <- c("MEblue", "MEpurple", "MEgreen", "MEpink")

sig_module_eigengenes <- module_eigengenes[, !(colnames(module_eigengenes) %in% unsig_modules)]

hubs_genes_correlation <- function(sig_module,colors, gene_corr) {
  hubGenesList <- list()
  
  for (module in colnames(sig_module)) {
    module_name <- gsub('ME', '', module)
    
    module_interest <- which(colors$unlist.moduleColors. == module_name)
    
    module_interest_genes <- gene_corr[module_interest, module]
    
    sorted_genes <- sort(module_interest_genes, decreasing = TRUE)
    top_5_percent_index <- ceiling(length(sorted_genes) * 0.1)
    hubGenes <- names(sorted_genes[1:top_5_percent_index])
    
    hubGenesList[[module_name]] <- hubGenes
  }
  
  return(hubGenesList)
}





hubgenes_corr= hubs_genes_correlation(sig_module_eigengenes,moduleColors,module.genes.corr)

hubs=as.data.frame(unlist(hubgenes_corr))

write.csv(hubs,'10%_hubs_corr.csv')


