import matplotlib.pyplot as plot
import networkx as nx
import numpy as np
import sys
import copy


#Implementation de l'algo de Ford-Fulkerson pris d'ici: https://www.programiz.com/dsa/ford-fulkerson-algorithm
class Graph:

    def __init__(self, graph):
        self.graph = graph
        self. ROW = len(graph)


    # Using BFS as a searching algorithm 
    def searching_algo_BFS(self, s, t, parent):

        visited = [False] * (self.ROW)
        queue = []

        queue.append(s)
        visited[s] = True

        while queue:

            u = queue.pop(0)

            for ind, val in enumerate(self.graph[u]):
                if visited[ind] == False and val > 0:
                    queue.append(ind)
                    visited[ind] = True
                    parent[ind] = u

        return True if visited[t] else False

    # Applying fordfulkerson algorithm
    def ford_fulkerson(self, source, sink):
        parent = [-1] * (self.ROW)
        max_flow = 0

        while self.searching_algo_BFS(source, sink, parent):

            path_flow = float("Inf")
            s = sink
            while(s != source):
                path_flow = min(path_flow, self.graph[parent[s]][s])
                s = parent[s]

            # Adding the path flows
            max_flow += path_flow

            # Updating the residual values of edges
            v = sink
            while(v != source):
                u = parent[v]
                self.graph[u][v] -= path_flow
                self.graph[v][u] += path_flow
                v = parent[v]

        return max_flow


#Cree la matrice d'adjacence du graph a partir du fichier
def create_matrice_adjacence():
    file = open(sys.argv[1], "r")
    line = file.readline()
    line_elements = line.split(';')

    #Contiendera une liste de nos arcs:
    arcs = []

    matrice_taille = int(line_elements[0])
    nbre_arc_a_enlever = int(line_elements[1])

    #Cree une matrice d'adjacence de taille matrix_size*matrix_size:
    matrice_adjacence = [[0]*matrice_taille for _ in range(matrice_taille)]

    #On skip la premiere ligne:
    line = file.readline()

    #Index de la source du graph (correspond au premier noeud du fichier)
    index_source = int(line.split(';')[0])

    #Pour chaque ligne du fichier:
    while line:
        line_elements = line.split(';')
        #Enlever les \n qui sont en trop lors de la lecture des lignes
        line_elements = [line_elements.strip().replace('\n', '') for line_elements in line_elements]

        #Pour chaque element entre deux ';' dans la ligne:
        for index, element in enumerate(line_elements):

            #Ici on selectionne le premier nombre (qui correspond au sommet qui envoie des flots):
            if index == 0:
                sommet = int(line_elements[0])
                continue
            
            autre_sommet = int(element.split("(")[0])
            poids = int(element.split("(")[1].replace(")", ""))

            matrice_adjacence[sommet][autre_sommet] = poids

            #Liste d'arcs sans doublons:
            arcs.append((sommet, autre_sommet))

        #Index du puits (correspond au dernier noeud du fichier)
        index_puits = int(line.split(';')[0])

        line = file.readline()
    #Debug:
    #for ligne in matrice_adjacence:
        #print(ligne)
    file.close()
    
    return matrice_adjacence, arcs, nbre_arc_a_enlever,index_source, index_puits

#Visualise le graph et le sauvegarde sous le nom: "graph.png"
def save_graph(matrice_adjacence):
    G = nx.DiGraph(np.array(matrice_adjacence))
    #layout = nx.fruchterman_reingold_layout(G)
    layout = nx.spring_layout(G, seed=0)
    nx.draw(G, layout, node_size=1000, with_labels=True,font_weight='bold',font_size=15)
    labels = nx.get_edge_attributes(G,'weight')
    nx.draw_networkx_edge_labels(G,pos=layout,edge_labels=labels)
    #plot.show()
    plot.savefig("Graph.png")

#Affiche et retourne le flot max qui peut passer par le graph (on precise quel est le sommet source et le sommet puits)
def calculer_flot_max(graph, source, sink):
    g = copy.deepcopy(graph)
    g = Graph(g)
    flot_max = g.ford_fulkerson(source, sink)
    return flot_max

def algo_recherche_flot_minimal(matrice_adjacence, k, source, puits, arcs):
    
    compteur_arcs_restants = k
    arcs_initials = arcs
    matrice_initial = copy.deepcopy(matrice_adjacence)
    meilleur_flot_trouve = calculer_flot_max(matrice_initial, source, puits)
    meilleur_matrice_trouve = copy.deepcopy(matrice_initial)
    meilleurs_arcs_enleves = []

    def algo_retour(matrice_actuelle, arcs_restants: list, k_restants: int):

        nonlocal arcs_initials, meilleur_flot_trouve, meilleur_matrice_trouve, meilleurs_arcs_enleves

        #On genere les futurs matrices a partir de matrice_actuelle
        prochains_graphs_a_explorer = []
        for arc in arcs_restants:
            sommet, autre_sommet = arc
            prochain_graph = copy.deepcopy(matrice_actuelle)
            prochain_graph[sommet][autre_sommet] = 0
            
            arcs_prochain_graph = copy.deepcopy(arcs_restants)
            arcs_prochain_graph.remove((sommet, autre_sommet))

            prochains_graphs_a_explorer.append((prochain_graph, arcs_prochain_graph))
            prochains_graphs_a_explorer

        #Pour chaque matrice, on verifie si on continue d'explorer, si oui: retourne True
        for matrice_a_explorer, arcs_actuels in prochains_graphs_a_explorer:           
            flot_actuelle = calculer_flot_max(matrice_a_explorer, source, puits)

            ########## DEBUG ###########:
            debug_enleves = [x for x in arcs_initials if x not in arcs_actuels]
            print()
            print("Pour cette matrice: ")
            print("Flot max: %d" % flot_actuelle)
            print("Arcs enleves:", debug_enleves)        
            for line in matrice_a_explorer:
                print(line)


            if(flot_actuelle < meilleur_flot_trouve):
                print ("Puisque le flot actuelle est inferieure au flot max: %d" % meilleur_flot_trouve)
                print ("On explore plus dans cette matrice")
                meilleur_flot_trouve = flot_actuelle
                meilleur_matrice_trouve = matrice_a_explorer
                meilleurs_arcs_enleves = arcs_actuels
                k_restants -= 1
                if k_restants <= 0:
                    continue

                algo_retour(matrice_a_explorer, arcs_actuels, k_restants)


    algo_retour(matrice_initial, arcs_initials, compteur_arcs_restants)
    return meilleur_matrice_trouve, meilleur_flot_trouve, meilleurs_arcs_enleves




matrice_adjacence, arcs, nbre_arcs_a_enlever, source, puits = create_matrice_adjacence()

meilleur_matrice, meilleur_flot, meilleur_arcs = algo_recherche_flot_minimal(matrice_adjacence, nbre_arcs_a_enlever, source, puits, arcs)

meilleur_arcs = [x for x in arcs if x not in meilleur_arcs]

print(meilleur_arcs, meilleur_flot)

#print(backtrack_reduce_flot_max(matrice_adjacence, 1, source, puits, arcs))