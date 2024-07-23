library(rat2302.db)
library(annotate)
gene_data=read.csv('10%_hubs_corr.csv')
#remove x
gene_data$unlist.hubgenes_corr.=gsub('X','',gene_data$unlist.hubgenes_corr.)

#probe id
probe_id=c(gene_data$unlist.hubgenes_corr.)

genes <- select(rat2302.db, probe_id, c("SYMBOL","GENENAME","ENTREZID"))



repeated_entrez_id <- as.data.frame(table(genes$ENTREZID))
idx_enter_id <- repeated_entrez_id[repeated_entrez_id$Freq > 1, ]




dub=duplicated(genes$PROBEID)
f=genes[!duplicated(genes$PROBEID), ]

df_genes <- distinct(genes,ENTREZID,SYMBOL,GENENAME)

enter_id=df_genes$ENTREZID
enter_id=as.data.frame(enter_id)
write.csv(enter_id,'Genes.csv')
#remove duplicates 
read.csv("/Users/miladvujude/Desktop/Master/Toxicogemic_Data/GENE_selection/WCGNA/Genes.csv")

#######
library(ggraph)
library(clusterProfiler)
library(org.Rn.eg.db)
library(enrichplot)
library(DOSE)




go_enrichment <- enrichGO(gene          = genes_non_repated_id$ENTREZID,
                          OrgDb         = org.Rn.eg.db,
                          keyType       = 'ENTREZID',
                          ont           = "ALL",
                          pAdjustMethod = "BH",
                          pvalueCutoff  = 0.01,
                          qvalueCutoff  = 0.05) 


barplot(go_enrichment , showCategory = 15)
dotplot(go_enrichment , showCategory = 15)

david <- enrichDAVID(gene = genes_non_repated_id$ENTREZID,
                     idType = "ENTREZ_GENE_ID",
                     annotation = "GOTERM_BP_FAT",
                     david.user = "my_mail")

