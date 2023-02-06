---
title: 'Expressivité des GNNs'
date: 2022-09-26
permalink: images/post_expressivity/
tags:
  - expressivity
  - gnn

---


# Expressivité des GNN


# GNN et expressivité

Les Graph Neural Networks sont les modèles états de l’art pour de l’apprentissage de représentation sur les graphes. Les architectures type ***************message passing*************** se sont majoritairement imposées dans ce domaine par leur simplicité et leurs performances sur les tâches impliquant des graphes. 

Cependant contrairement à leurs semblables dans la famille des réseaux de neurones (CNN,MLP,Transformers), leur expressivité est limitée par le test de Weisfeiler-Lehman. Dans cette note de blog on va voir tout d’abord ce qu’est l’algorithme de Weisfeiler-Lehman, pourquoi il est essentiel de le comprendre afin de designer des modèles plus performants et de dépasser les limites des architectures standards de GNN. 

On va voir ensuite comment les récents modèles GNN arrivant à dépasser les bornes d’expressivité du test 1-WL, et pour finir on va parler de l’expressivité des modèles de GNN sur graphes dynamiques, domaine très peu étudié à ce jour. 

## Expressivité et test de Weisfeiler-Lehman.

### 1-WL Test

****************************************[Les figures suivantes ont été emprunté au blogpost sur l’expressivité des GNNs de M. Bronstein](https://towardsdatascience.com/expressive-power-of-graph-neural-networks-and-the-weisefeiler-lehman-test-b883db3c7c49)****************************************

Le 1-WL Test est un algorithme en temps polynomial mis au point par Weisfeiler et Lehman en 1968 afin de déterminer si deux graphes sont isomorphes ou non. Des graphes isomorphes signifient que leurs structures sont identiques. 

Le test consiste en la recoloration itérative des nœuds d’un graphe. 

Pour chaque nœud, on agrège les couleurs de son voisinage puis à l’aide d’une fonction de hachage inductive on recolore le nœud courant. On répète ce processus jusqu’à convergence (La couleur de chaque nœud n’évolue plus par rapport à l’itération précédente).

Si deux graphes sont isomorphes alors à la fin du processus leurs colorations seront identiques. 

![Untitled](Untitled.png)

 Cet algorithme arrive à distinguer les structures de graphes dans **quasiment toute les situations** sauf certaines exceptions comme les graphes réguliers par exemple. 

> *Le test WL est basé sur la recoloration itérative des graphes (Une "couleur" en théorie des graphes désigne une étiquette discrète des nœuds), en commençant par tous les nœuds de couleur identique. À chaque étape, l'algorithme agrège les couleurs des nœuds et de leurs voisinages en les représentant sous forme de multisets, puis il hache les multisets de couleurs agrégés en de nouvelles couleurs uniques. L'algorithme s'arrête lorsqu'il atteint une coloration stable. Si, à ce moment-là, les colorations des deux graphes diffèrent, les graphes sont considérés comme non isomorphes. En revanche, si les colorations sont les mêmes, les graphes sont probablement (mais pas nécessairement) isomorphes. En d'autres termes, le test WL est une condition nécessaire mais insuffisante pour l'isomorphisme des graphes. Il existe des graphes non isomorphes pour lesquels le test de WL produit des colorations identiques et les considère donc comme "possiblement isomorphes" ; on dit que le test échoue dans ce cas. Un tel exemple est illustré dans la figure suivante :*
> 

![Untitled](Untitled%201.png)

### How Powerful are GNN (2018)

How Powerful are GNN a montré que les architectures standards de message passing (GCN,GAT,SAGE) sont limités en expressivité par le test de Weisfeiler Lehman, de plus dans la plupart des cas ces GNNs ont une capacité bien moindre au 1-WL test pour discerner les structures de graphes.

Le bornage signifie que les structures de graphes indiscernable par ce test ne le seront également pas par les MP-GNN. Dans l’article How Powerful are GNN les auteurs ont ensuite mis au point le MP-GNN ayant la plus haute expressivité et comparable au 1-WL test, le modèle ******GIN******.

Pour qu’une architecture soit aussi expressive que le 1-WL Test il faut poser certaines conditions sur l’agrégation du voisinage et sur la fonction *******readout.******* 

**********************La fonction readout agrège l’ensemble des embeddings des noeuds afin de produire une représentation du graphe.********************** 

En effet, un GNN expressif ne doit jamais représenter de la même manière deux voisinages différents, il nous faut donc une fonction d’agrégation du voisinage ******************injective******************. De même pour la fonction ***readout*** générant la représentation globale du graphe. 

**L’objectif est le suivant:** 

$$
\text{Soit } G_{1} \text{ et }   G_{2} \text{ deux graphes non isomorphes alors un GNN } \mathcal{G} \rightarrow \mathbb{R}^{d} \text{ représente } G_{1} \text{ et } G_{2} \text{ différement.}
\\
 
\text{Le test de Weisfeiler-Lehman doit également déterminer que } G_{1} \text{ et } G_{2} \text{ sont isomorphes.} 
$$

 Les auteurs introduisent le théorème suivant:

$$
\text{Soit } \mathcal{A}: \mathcal{G} \rightarrow \mathbb{R}^{d} \text{ un GNN, } \mathcal{A} \text{ est une application qui pour n'importe quels graphes } G_{1} \text{ et } G_{2} \\
\text{ déclarés comme non isomorphes par le WL-test, les représentent de manière différente si les}\\  
\text{ conditions suivantes respectent: }
\\

\\
\text{a) } \mathcal{A} \text{ agrège et met à jour les représentations des noeuds comme suivant : }\\ 
h_{v}^{(k)} = \phi(h_{v}^{(k-1)},f(\{(h_{u}^{(k-1)} : u \in \mathcal{N}(v)\}))
\\
\text{b) La fonction \textit{readout}  opérant sur le \textit{multiset} des représentations de noeuds est injective.} 
$$

### Dépasser le 1-WL Test

******************Test k-WL******************

Les recherches récentes sur les GNNs ont essayé de dépasser la borne d’expressivité du test 1-WL. 

En effet, le test 1-WL est limité et ne permet pas de prendre en compte des structures simples de graphe comme les triangles clos ou les motifs circulaires ([Can Graph Neural Networks Count Substructures?](https://arxiv.org/abs/2002.04025))**.**

Le test WL peut être généralisé à une hiérarchie de tests d'ordre supérieur, connue sous le nom de test k-WL. Cette hiérarchie a été utilisée pour caractériser la puissance d'expressivité des GNNs et pour inspirer la conception d'architectures de nouveaux GNNs.

Dans l’algorithme k-WL (k>1) on ne cherche plus la coloration d’un noeud, mais la coloration d’un k-tuple de nœuds. 

Par exemple dans le 2-WL test on considérera non plus un noeud et son voisinage mais une paire de noeud. On n’agrège plus la coloration des noeuds voisins (cas 1-WL) mais la coloration des triangles voisins. 

Cela permet au test de décrire des motifs plus complexes, et donc de dépasser l’expressivité du 1-WL test. 

![Capture d’écran de 2023-01-25 16-57-24.png](Capture_dcran_de_2023-01-25_16-57-24.png)

Dans l’exemple ci dessus, le 1-WL test détecte une isomorphie tandis que le test 3-WL distingue les deux graphes. 

**************************************************************************************************Bien que plus expressif, la complexité du test k-WL est beaucoup plus grande, du fait de considérer chaque k-tuples de nœuds du graphe, y compris ceux qui ne sont pas adjacents.**************************************************************************************************  

La vidéo ci dessous permet de mieux comprendre le test WL et de mieux visualiser les cas en plus haute dimension. 

[https://www.youtube.com/watch?v=YHeeAB_XRS4&list=PLeKJtN5YzWyPoNyqL34y5Jrag7aqgQRn5&index=10](https://www.youtube.com/watch?v=YHeeAB_XRS4&list=PLeKJtN5YzWyPoNyqL34y5Jrag7aqgQRn5&index=10)

Une playlist complète sur l’isomorphisme des graphes, l’algorithme de Weisfeiler-Lehmann ainsi que les graphes transformers est disponible [ici.](https://www.youtube.com/playlist?list=PLeKJtN5YzWyPoNyqL34y5Jrag7aqgQRn5) 

- Ressources supplémentaires sur l’algorithme WL ainsi que ses adaptations en Machine Learning
    - [**Weisfeiler and Leman go Machine Learning: The Story so far**](https://arxiv.org/abs/2112.09992)
    - [**A survey on the expressive power of graph neural networks**](https://arxiv.org/abs/2003.04078)
    - [**A Short Tutorial on The Weisfeiler-Lehman Test And Its Variants**](https://arxiv.org/abs/2201.07083)

**********K-GNN**********

Des contributions comme ********************************[WL Go Neural : Higher Order GNN](https://arxiv.org/abs/1810.02244)********************************  ou [*Provably Powerful Graph Networks](https://arxiv.org/abs/1905.11136)* arrivent à concevoir des k-GNN en se basant sur le test k-WL et ainsi atteindre la même expressivité théorique. Cependant ces techniques ont une grande complexité en mémoire et en temps et sont souvent inapplicable sur des gros graphes.

 

![Capture d’écran de 2023-01-25 16-54-27.png](Capture_dcran_de_2023-01-25_16-54-27.png)

                                          ********Source:********  ********************************[WL Go Neural : Higher Order GNN](https://arxiv.org/abs/1810.02244)********************************

**************************************Méthodes de GNN sur sous graphes**************************************

La décomposition d’un graphe en plusieurs sous graphes traitées par un GNN améliore grandement l’expressivité des modèles. Bouritsas et al. présente leur contribution d’une méthode rapide et plus expressive que le 1-WL test:  **[Improving Graph Neural Network Expressivity via Subgraph Isomorphism Counting](https://arxiv.org/abs/2006.09252)** (ICLR 2021)

Une note sur ce papier est détaillée dans ce post de M.Bronstein. 

[Beyond Weisfeiler-Lehman: using substructures for provably expressive graph neural networks](https://towardsdatascience.com/beyond-weisfeiler-lehman-using-substructures-for-provably-expressive-graph-neural-networks-d476ad665fa3)

L’idée est de faire en sorte de traîter les messages différement et de manière explicite en fonction de la structure locale du graphe. Les messages sont traitées dans des fonctions additionnelles décrivant la structure associée à chaque noeud. Ainsi les messages provenant de structures différentes ne seront pas agrégés de la même manière. Pour ce faire il sera nécessaire de compter certaines sous structures, le choix de ces sous structures permet d’injecter un biais inductif qui peut grandement améliorer l’apprentissage pour certaines tâches (par ex. compter les triangles clos dans un réseau social ou les structures circulaires dans les interactions de protéines). 

Leur modèle ******GSN****** (Graph Substructure Networks) permet de théoriquement dépasser l’expressivité du test 1-WL tout en gardant la notion de localité et la même complexité ************************************************(pendant l’entraînement)************************************************ que les architectures de Message Passing contrairement aux méthodes k-GNN dont on a discuté précedemment. 

Méthode:

Le coeur du modèle GSN est le préprocessing construisant les features de noeuds encodant les sous structures prédifinis dans le graphe $G$. 

Il faut au préalable définir un ensemble de structures que l’on cherche à identifier dans notre graphe:                                         

$\mathcal{H} = \{H_{1},H_{2},...,H_{k}\}$

L’ensemble $\mathcal{H}$  contient des sous structures comme par exemple une 3-clique ou un cycle d’une longueur définie. 

Pour chaque $H$ trouve l’ensemble de ses sous graphes isomorphes dans $G$ que l’on note $G_{S}$.

Pour chaque noeuds $v \in V_{G_{S}}$ on infère son rôle dans $H$ en obtenant l’orbite de son application $f(v) \text{ dans } H \text{  Orb}_{H}(f(v)).$ En comptant toute les apparences possible des différentes orbites dans $v$ on obtient la feature structurelle de $v, \mathbf{x}_{H}^{V}$. 

La complexité de ce preprocessing est dans le pire des cas $\mathcal{O}(n^{k})$ avec $k$ le cardinal de l’ensemble $\mathcal{H}$. 

![Capture d’écran de 2023-01-26 14-41-27.png](Capture_dcran_de_2023-01-26_14-41-27.png)

                                                                       ************Source: GSN************ 

Dans la figure (partie gauche) ci dessous on voit qu’au voisinage du noeud courant (bleu) se trouve un motif 3-cycle (rouge) 4 noeuds (jaune) et trois chemins de longueur 3. 

C’est le même principe pour la partie droite de la figure où l’on cible non plus un noeud mais une arête. 

Le message passing est définie comme suivant: 

![Capture d’écran de 2023-01-26 14-46-11.png](Capture_dcran_de_2023-01-26_14-46-11.png)

$M^{t+1}$ est une fonction d’agégation des messages (MLP, attention…) $UP^{t+1}$ est la fonction de mise à jour de la représentation courante du noeud.

Ce sont les mêmes opérations que dans un MP-GNN classique. Les deux variantes ************GSN-e************ et **********GSN-v********** représentent dans le premier cas le comptage des sous structures autour d’une arête cible et dans le second cas autour d’un noeud cible. 

******************GSN est un modèle GNN-message passing augmenté à l’aide de features préalablement construites encodant des sous structures dans le voisinage local d’un noeud ou d’une arête.******************

Le choix des sous structures est crucial pour une bonne généralisation du modèle. Il est important d’avoir une bonne connaissance à priori de la distribution du graphe (privilégié les graphlets dans les interactions de protéines, et les cliques et triangles dans les réseaux sociaux par exemple). Le choix des sous structures est une question ouverte. 

Une place importante est consacré à l’expressivité du GSN dans leur article, ils prouvent que le GSN est ****************au moins**************** aussi expressif que le 1-WL test, et qu’en fonction du choix des sous structures, ils arrivent à atteindre une expressivité égale au test k-WL. 

![Untitled](Untitled%202.png)

                                                                         *[Source : blog Bronstein](https://towardsdatascience.com/beyond-weisfeiler-lehman-using-substructures-for-provably-expressive-graph-neural-networks-d476ad665fa3)*

### Depuis le modèle GSN

***************Basé sur le blogpost de M.Bronstein :*************** 

[Using subgraphs for more expressive GNNs](https://towardsdatascience.com/using-subgraphs-for-more-expressive-gnns-8d06418d5ab)

Le modèle GSN a conduit à de nombreuses contributions dans le domaine des GNNs basés sur les sous graphes. 

**************Dropout**************

Il a été montré que l’on peut exploiter des sous-graphes et ainsi augmenter l’expressivité des GNNs à l’aide du dropout comme les modèles ********[DropGNN](https://arxiv.org/pdf/2111.06283.pdf) (NIPS 2021)* ou ********[DropEdge](https://openreview.net/pdf?id=Hkx1qkrKPr) (ICLR 2020).* En retirant avec une certaine probabilité des noeuds de notre graphe pendant le message passing et en répétant ce dropout un certains nombre de fois on peut obtenir plusieurs embeddings de sous graphe au voisinage d’un noeud. 

Cette méthode permet de ne pas devoir sélectionner au préalable des sous structures.

******************************************GNN et reconstruction****************************************** 

[*Reconstruction for Powerful Graph Representations](https://arxiv.org/abs/2110.00577) (NIPS 2021)* est un modèle se basant sur la conjecture de reconstruction d’un graphe. 

> *De manière informelle, la conjecture de reconstruction en théorie des graphes dit que les graphes sont déterminés de manière unique par leurs sous-graphes.*
> 

Les GNN de reconstruction k proposés dans ce modèle appliquent un MP-GNN à chacun des sous-graphes induits de taille k et additionnent les embeddings résultants. Comme ces sous-graphes peuvent être trop nombreux , il faut recourir à l'échantillonnage (similaire aux GNN Dropout).

******************Stars to Subgraphs******************

[*From Stars to Subgraphs: Uplifting Any GNN with Local Structure Awareness](https://arxiv.org/abs/2110.03753) (ICLR 2022)*

![Untitled](Untitled%203.png)

Dans les MP-GNN et dans 1-WL, l’agrégation est une structure en étoile. Pour étendre cette représentation limité Zhao et al. utilise les ****************rooted subgraphs.**************** 

Les ****************rooted subgraphs**************** en plus de capturer les informations du voisinage comme le motif en étoile (figure gauche) on capture également sa connectivité (figure droite). 

Cette façon d’agréger le voisinage d’un noeud est appelée ******Sous-graphes 1-WL******. Dans cet article ils montrent que c’est au moins aussi expressif que l’algorithme 3-WL. 

Les GNN basés sur les Sous-graphes 1-WL encode tout les sous graphes avec un MP-GNN. 

En plus des architectures de dropout et les architectures de reconstruction, cette méthode a l’avantage d’agréger des sous graphes impliquant plusieurs fois le même noeud. 

**********************************************Sous graphe équivariant**********************************************

[*Equivariant Subgraph Aggregation Networks](https://arxiv.org/abs/2110.02910) ESAN (ICLR 2022)*

Plutôt que d’encoder des *********multisets********* de noeuds comme dans 1-WL ou les MP-GNN, ils proposent d’encoder un multisets de sous graphes. 

![Capture d’écran de 2023-01-26 16-38-25.png](Capture_dcran_de_2023-01-26_16-38-25.png)

                                                                           *******************Source : ESAN******************* 

Pour constituer ces multisets de sous graphes ils proposent 4 politiques de séléctions : 

- Suppression de noeud
- Suppression d’arêtes
- 2 variantes des égo-subgraphs (voir paragraphe précédent).

Leur modèle est un framework equivariant pour générer et traiter ces sous graphes. 

Pourquoi équivariant ? 

![Untitled](Untitled%204.png)

À partir du graphe en entrée (figure gauche) on peut avec la méthode de suppression d’arête générer 3 sous graphes (figure du milieu), ce qui conduit à 3 représentation tensorielle différentes, pourtant il ne s’agit que d’une permutation du même motif de sous graphe. 

Le framework equivariant a pour but de générer la même représentation pour chacune de ces trois représentations à l’aide des groupes de symétries, ESAN est basé sur les travaux présentés par [Maron et al.](https://arxiv.org/abs/2002.08599) (apprentissage de motif symétrique) 

 

![Untitled](Untitled%205.png)

ESAN est composé de deux encodeurs (orange et jaune), le premier encode les structures de sous graphes en parallèle à l’aide de réseau siamois, le second ********************information sharing******************** est une agrégation (somme) des différents tenseurs de représentations des sous graphes. 
L’encodeur orange est donc une généralisation des modèles du type Dropout ou Reconstruction, et le module de partage d’information est une généralisation de STAR. 

********************************************************************************************************************Un des enjeux de toute ces architectures de sous-graphes est l’échantillonnage, en effet l’espace des sous graphes autour d’un noeud cible est souvent extrêmement grand. Il est donc important soit de prédéfinir les structures (ce qui nécessite une connaissance à priori sur la distribution du graphe) soit d’échantillonner (de manière plus ou moins intelligente) les structures de sous graphes.********************************************************************************************************************

********************************************[Plus récemment (2022)](https://towardsdatascience.com/graph-ml-in-2023-the-state-of-affairs-1ba920cb9232#8e6c)******************************************** 

> *Dans le domaine des architectures plus expressives (que 1-WL), les GNN à sous-graphes sont la plus grande tendance. Parmi celles-ci, trois approches se distinguent :

 1️⃣ [Subgraph Union Networks (SUN) de Frasca, Bevilacqua, et al.](https://arxiv.org/abs/2206.11140) qui fournissent une analyse complète de l'espace de conception et de l'expressivité des GNN à sous-graphes, montrant qu'ils sont limités par le test 3-WL ;

 2️⃣ [Ordered Subgraph Aggregation Networks](https://arxiv.org/abs/2206.11168) (OSAN) de Qian, Rattan, et al conçoivent une hiérarchie de sous graphes améliorés par des GNNs (k-OSAN) et constatent que les k-OSAN sont incomparables aux k-WL mais sont strictement limités par (k+1)-WL. Une partie particulièrement cool d'OSAN est l'utilisation de [Implicit MLE (NeurIPS'21)](https://arxiv.org/abs/2106.01798), une technique d'échantillonnage discret différentiable, pour l'échantillonnage des sous-graphes ordonnés.
 ️
3️⃣ [SpeqNets](https://arxiv.org/abs/2203.13913) de Morris et al. conçoit une hiérarchie permutationnelle équivariante de réseaux de graphes qui trouve un équilibre entre la scalabilité et l'expressivité.

 4️⃣ [GraphSNN](https://openreview.net/pdf?id=uxgg9o7bI_3) de Wijesinghe et Wang dérive des modèles expressifs basés sur le chevauchement des isomorphismes de sous-graphes et des isomorphismes de sous-arbres.*
> 

## Expressivité des GNNs dynamiques :

## Provably Expressive Temporal Graph Networks (Neurips 2022)

### Contexte:

Ce papier étudie l’expressivité des TGN (Temporal Graph Networks) en classifiant deux familles de TGN; la première étant les TGN agrégeant des marches aléatoires temporelles **WA-TGN** (CAW), la seconde contient les modèles construit sur du message passing ************MP-TGN************ (TGN, TGAT, JODIE…), dans les deux cas on s’intéresse au Graphe dynamique continue (**CTDG**), les DTDG pouvant être immédiatement converti en CTDG. 

***********************************************Rappel des architectures SOTA des TGN sur CTDG.***********************************************

|                    MP-TGN |                   WA-TGN |
| --- | --- |
| https://arxiv.org/abs/1803.04051: Les approches basées sur les processus ponctuel et temporels (TPP), dans lesquelles les processus de points temporels sont paramétrés par un réseau de neurones (RNN). | https://arxiv.org/abs/2101.05974
De la famille des WA-TGN, CAW-N génère des marches aléatoires anonymes (anonyme important pour que le modèle soit inductif) et causal. À partir d’une arête cible (u,v), on génère une marche aléatoire pour déterminer les noeuds responsable dans la création de ce lien. Cette marche est encodé à l’aide d”un RNN constituant la représentation du lien uniquement .  |
| https://arxiv.org/abs/1908.01207: Approche RNN, les noeuds sont màj à l’aide d’un modèle de time serie. Cas de graphe biparti, deux RNN (utilisateurs et items). Une projection temporelle est utilisée pour màj la mémoire d’un noeud. Si le noeud n’a été impliqué dans aucun événement depuis un certain temps, sa mémoire est considérée comme périmée. 
 |  |
| https://arxiv.org/abs/2002.07962: Première architecture utilisant du Time Embedding. Durant l’agrégation des messages on utilise un time embedding (Time2Vec) pour représenter l’écoulement du temps entre le moment présent t et le moment de la dernière interaction avec le noeud voisin.
La fonction d’agrégation est similaire à GAT (utilisation de facteur d’attention dans l’agrégation des voisins). 
L’ensemble des couches GNN agrégés Z(t) est ensuite projeté linéairement pour obtenir les “query” “key” et “value”, similaire aux transformers.  |  |
| https://arxiv.org/abs/2006.10637Framework générique de message passing. En plus de TGAT, TGN-atn utilise un module de mémoire pour le MP et la màj des noeuds. 
Durant l’agrégation des messages on utilise également Time2Vec. |  |

L’objectif de cette recherche est d’étendre les études de l’expressivité des GNN statiques (GIN: How Powerful are GNN, les connections aux algorithmes basés sur le test WL…)  aux graphes temporelles en définissant un test temporelle-1-WL. 

À l’issue de l’étude de l’expressivité des GNN temporelles, les auteurs proposent une nouvelle architecture **PINT**. En exploitant un message passing temporelle injectif ainsi que des *****************************relatives positional features***************************** ils montrent des gains important par rapport aux archi existantes en ayant une expressivité comparable au test temporelle 1-WL. 

![Capture d’écran de 2023-01-27 14-44-09.png](Capture_dcran_de_2023-01-27_14-44-09.png)

### Isomorphisme dans les graphes temporelles

L’expressivité des modèles TGNs est évaluée en fonction de leurs capacités à différencier des structures non isomorphes dans le graphe. 
Deux nœuds n’ayant pas la même structure locale et n’ayant pas établi des connections avec son voisinage au même instant $t$ ne doivent pas avoir la même représentation. 

 

Afin de mesurer l’expressivité d’un MP-TGN on évalue leurs capacités à représenter de manière différente dans l’espace des embeddings deux noeuds distincts. 

$u\text{ et } v \text{ sont différents alors } h_{u}^{(L)}(t) \neq h_{v}^{(L)}(t).$

Le calcul de représentation d’un noeud $v$ à un temps $t$  peut être décrit avec un TCT (**************************Temporal Computation Tree**************************) que l’on note $T_{v}(t)$.

 

![Capture d’écran 2023-01-30 à 11.38.27.png](Capture_decran_2023-01-30_a_11.38.27.png)

Le TCT est un arbre de calcul induit à partir d’un nœud source. 

À gauche, on a un graphe temporelle, les couleurs représentent les ********features******** de noeuds, $t_{1},t_{2}$  sont les instants de connections entre les noeuds. À droite, on a les TCT d’une profondeur 2 pour les noeuds $u$ et $v.$ 

Si un MP-TGN d’une profondeur $L$  est capable de donner une représentation différente pour $u$ et $v$, alors $T_{u}(t) \text{ et } T_{v}(t)$ ne sont pas isomorphes.

Dans l’exemple de la figure précédente, les modèles comme TGN-att ou TGAT ne sont pas capable de distinguer $T_{u}(t)$ et $T_{v}(t)$. En effet ces deux modèles utilisent la moyenne des messages comme fonction d’agrégation, comme on a la même proportion de ********features******** dans les deux cas, les deux modèles donneront la même représentation à $u$  et $v.$

![Capture d’écran 2023-01-30 à 12.10.35.png](Capture_decran_2023-01-30_a_12.10.35.png)

Des structures simples telles que les cycles ou les cliques ne sont pas modélisable par les MP-TGN ou les WA-TGN actuel, et ce même dans le cas où les MP-TGN ont une expressivité similaire au test temporel 1-WL.

Dans le papier, une place est accordée à l’expressivité des MP-TGN comprenant un module de mémoire stipulant que si la profondeur du TCT est inférieur au nombre de couche $L$ , ce module n’améliorera pas l’expressivité du modèle. 

**********************************Extension du 1-WL au cas temporel.********************************** 

1-WL est un test d’isomorphisme sur des graphes statiques se basant sur la recoloration itérative des nœuds du graphe, à la fin du processus si deux graphes ont la même colorations alors ils sont (**************************probablement**************************) isomorphes.

Il est possible d’étendre cet algorithme aux graphes temporels, il suffit de considérer le graphe dynamique comme un multigraphe (plusieurs liens possible entre chaque noeuds). 

### Modèle PINT

Les auteurs présent PINT (******po****sition-encoding ****in****jective **t**emporal graph net*). Ce modèle est construit sur deux composantes: 

1. L’agrégation injective des messages ainsi que la mise à jour des représentations dans un contexte temporel. 
2. Générer des ********features******** positionnel à partir des arbres de calcul TCT. 

******************************************Agrégation injective.****************************************** 

Comme dans GIN, le meilleur moyen pour maximiser l’expressivité des modèles MP-TGN est d’utiliser des fonctions injective d’agrégation. Dans le cas des graphes dynamiques il est important de considérer à quel instant les liens ont été établi entre deux noeuds. Un biais inductif courant est de considérer avec plus d’importances les événements récents (TGN,TGAT). 

$$
\tilde{h}_v^{(\ell)}(t)=\sum_{\left(u, e, t^{\prime}\right) \in \mathcal{N}(v, t)} \operatorname{MLP}_{\mathrm{agg}}^{(\ell)}\left(h_u^{(\ell-1)}(t) \| e\right) \alpha^{-\beta\left(t-t^{\prime}\right)}
$$

Comme fonction injective ils utilisent comme dans GIN, des MLP (approximateurs universels de fonctions continues). 

Pour donner plus d’importances aux événements récents PINT utilise une décroissance temporelle linéairement exponentielle, $\alpha^{-\beta(t-t^{'})}, \alpha \text{ et }\beta$ sont des hyper-paramètres. 

$$
h_v^{(\ell)}(t)=\operatorname{MLP}_{\mathrm{upd}}^{(\ell)}\left(h_v^{(\ell-1)}(t) \| \tilde{h}_v^{(\ell)}(t)\right)
$$

La fonction de MàJ doit également être injective pour maximiser l’expressivité, l’opérateur $\|$ dénote la concaténation de vecteurs. $h_{v}^{0} = s_{v}(t)$, le vecteur d’état du noeud $v$. 

************************Features******** positionnelles et relatives**

 Les auteurs proposent d’augmenter les vecteurs d’états $s$ de chaque noeuds avec des ********features******** positionnelles relatives. 

Le principe est de compter combien de marches temporelles d’une taille donnée existent entre deux noeuds (Combien de fois les noeuds apparaissent à des niveaux différents de TCT). 

On considère une matrice carrée $P$. Une ligne de cette matrice nous donne la position relative  de chaque noeuds par rapport au noeud cible. 

![Capture d’écran 2023-01-30 à 13.30.03.png](Capture_decran_2023-01-30_a_13.30.03.png)

********PINT******** augmenté des ********features******** positionnelles est strictement plus expressif que WA-TGN et MP-TGN. 

****************************Limite de PINT****************************

PINT est borné par le test temporel 1-WL. De même que T-1-WL, toutes les structures non isomorphes que T-1-WL déterminera comme isomorphes, PINT n’arrivera également pas à les distinguer. 

’

![Capture d’écran 2023-01-30 à 13.35.00.png](Capture_decran_2023-01-30_a_13.35.00.png)

La symétrie de ce graphe fait que PINT n’arrivera pas à donner une représentation différentes pour ces deux événements. Il est donc encore possible d’améliorer l’expressivité des modèles TGN. 

## Résultats

![Capture d’écran 2023-01-30 à 13.36.36.png](Capture_decran_2023-01-30_a_13.36.36.png)

Le protocole expérimentale ici utilisé est identique à TGN,TGAT,CAW… Il serait intéressant d’évaluer PINT dans le nouveau protocole présenté à NIPS 2022, **[Towards Better Evaluation for Dynamic Link Prediction](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwj66-rZrO_8AhVxiv0HHVyJC04QFnoECAoQAQ&url=https%3A%2F%2Farxiv.org%2Fabs%2F2207.10128&usg=AOvVaw3Gy6osPs-ORPj_Iv_h5t8Y).**