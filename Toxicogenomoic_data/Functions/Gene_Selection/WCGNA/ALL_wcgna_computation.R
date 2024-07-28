library(WGCNA)
library(data.table)
library(gridExtra)
library(ggplot2)
library(dplyr)
library(tibble)
library(devtools)
library(CorLevelPlot)
#read document and transform it 
###this data is toxicology data

data <- read.csv('gene_expression.csv')
#remove first column
rownames(data) <- data$X
data <- data[, -1]

##prepreation data for histogram
toxicology_ht=copy(data)
rownames(toxicology_ht) <- gsub('24 hr', '', rownames(toxicology_ht))

#
htree <- hclust(dist(toxicology_ht), method = "average")
par(mar = c(15, 4, 4, 2)) 
plot(htree,cex = 0.6)

#pick best pwoer for network constrcution
power <- c(c(1:20))
sft <- pickSoftThreshold(data,
                         powerVector = power,
                         networkType = "signed",
                         verbose = 5)


sft.data <- sft$fitIndices

# visualization to pick power

a1 <- ggplot(sft.data, aes(Power, SFT.R.sq, label = Power)) +
  geom_point() +
  geom_text(nudge_y = 0.1) +
  geom_hline(yintercept = 0.80, color = 'red') +
  labs(x = 'Power', y = 'Scale free topology model fit, signed R^2') +
  theme_classic()


a2 <- ggplot(sft.data, aes(Power, mean.k., label = Power)) +
  geom_point() +
  geom_text(nudge_y = 0.1) +
  labs(x = 'Power', y = 'Mean Connectivity') +
  theme_classic()


grid.arrange(a1, a2, nrow = 2)

#plot data
data[] <- sapply(data, as.numeric)

soft_power <- 5
temp_cor <- cor
cor <- WGCNA::cor

bwnet <- blockwiseModules(data,
                          power = soft_power,
                          maxBlockSize = 14000,
                          randomSeed = 1234,
                          TOMType = "signed",
                          deepSplit = 2,
                          mergeCutHeight = 0.25,  # Starting point, can be adjusted
                          numericLabels = FALSE,
                          verbose = 3
)


module_eigengenes= bwnet$MEs
# Print out a preview
head(module_eigengenes)

##save module_eigengenes


table(bwnet$colors)


moduleColors = bwnet$colors 

write.csv(as.data.frame(unlist(moduleColors)),'modules_18.csv')

# Plot the dendrogram and the module colors
plotDendroAndColors(bwnet$dendrograms[[1]], cbind(bwnet$unmergedColors, bwnet$colors),
                    c("unmerged", "merged"), dendroLabels = FALSE, hang = 0.03,
                    addGuide = TRUE, guideHang = 0.05)


pathology=read.csv('Pathological_finding_external_traits.csv')
rownames(pathology) <- pathology$X
pathology <- pathology[, -1]
nSamples <- nrow(data)

module.trait.corr <- cor(module_eigengenes, pathology, use = 'p')
module.trait.corr.pvals <- corPvalueStudent(module.trait.corr, nSamples)

#prepare data for heatmap
heatmap.data <- merge(module_eigengenes, pathology, by = 'row.names')

heatmap.data <- heatmap.data %>% 
  column_to_rownames(var = 'Row.names')


CorLevelPlot(heatmap.data,
             x = names(heatmap.data)[12:19],
             y = names(heatmap.data)[1:11],
             col = c("blue1", "skyblue", "white", "pink", "red"))



module.gene.mapping <- as.data.frame(bwnet$colors)
grey_genes=module.gene.mapping %>% 
  filter(`bwnet$colors` == 'grey') %>% 
  rownames()  %>% 
  as.data.frame()


write.csv(grey_genes,'grey_genes_18.csv')


write.csv(module_eigengenes , 'Soft_Power_Threshold/modules_18.csv')



