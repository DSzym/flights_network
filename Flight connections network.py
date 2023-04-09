import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statistics

G = nx.Graph()
# list nodes contains names of airports to analyse
nodes = ["London Heathrow Airport", "Charles de Gaulle International Airport", "Adolfo Suárez Madrid–Barajas Airport",
         "Leonardo da Vinci–Fiumicino Airport", "John F Kennedy International Airport",
         "Istanbul Airport", "Copenhagen Kastrup Airport", "Cairo International Airport",
         "Eleftherios Venizelos International Airport", "Narita International Airport", "Dubai International Airport",
         "King Fahd International Airport", "Frankfurt am Main Airport", "Warsaw Chopin Airport",
         "Sydney Kingsford Smith International Airport", "Suvarnabhumi Airport", "Dublin Airport",
         "Licenciado Benito Juarez International Airport", "Heraklion International Nikos Kazantzakis Airport",
         "Malta International Airport"]

# download data and name columns
data_airports = pd.read_csv('https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat',
                            names=('Airport ID', 'Name', 'City', 'Country', 'IATA', 'ICAO', 'Latitude', 'Longtitude',
                                   'Altitude', 'Timezone', 'DST', 'Tz database time zone', 'Type', 'Source'))

lista_danych_lotnisk = data_airports.values.tolist()
airport_dict = {}       # this dict will contain information of airports useful in later analysis

for row in lista_danych_lotnisk:
    for item in row:
        if item in nodes:
            airport_dict[row[4]] = row[1]

data_airlines = pd.read_csv('https://raw.githubusercontent.com/jpatokal/openflights/master/data/airlines.dat',
                            names=('Airline ID', 'Name', 'Alias', 'IATA', 'ICAO', 'Callsign', 'Country', 'Active'))
lista_danych_linii_lotniczych = data_airlines.values.tolist()

data_routes = pd.read_csv('https://raw.githubusercontent.com/jpatokal/openflights/master/data/routes.dat',
                          names=('Airline', 'Airline ID', 'Source airport IATA', 'Source airport ID',
                                 'Destination airport IATA', 'Destination airport ID', 'Codeshare', 'Stops',
                                 'Equipment'))

lista_danych_polaczen = data_routes.values.tolist()
lista_polaczen = []  # contains routes between airports from airport_dict

for row in lista_danych_polaczen:
    if row[2] in airport_dict.keys():
        if row[4] in airport_dict.keys():
            lista_polaczen.append([row[2], row[0], row[4], np.random.randint(10, 100)])
                                # Node_start, airline, Node_end, nr of flights

airline_form_routes = []
for row in lista_polaczen:
    airline_form_routes.append(row[1])
airline_form_routes = list(dict.fromkeys(airline_form_routes))

airlines_dict = {}
for row in lista_danych_linii_lotniczych:
    if row[3] in airline_form_routes:
        airlines_dict[row[3]] = [row[1], np.random.random_sample()]
        # [name, route colour]

# dict flight_by_airline categorize routes according to airlines
flight_by_airline = {}

for flight in lista_polaczen:
    name = flight[1]
    if name in flight_by_airline.keys():
        temp = flight_by_airline[name]
        temp.extend([flight])
        flight_by_airline[name] = temp
    else:
        flight_by_airline[name] = [flight]

airlines = flight_by_airline.keys()
items_to_del = []
for arln in airlines:  # in this loop is chosen which airlines will be analise later, rest are ignored
    list_of_flights = flight_by_airline[arln]
    if len(list_of_flights) < 3:
        items_to_del.append(arln)

for item in items_to_del:  # delete airlines with routhes they cover
    del flight_by_airline[item]

low_cost_airlines = ["Air Europa", "Air Berlin", "Air India Limited", "Air Malta", "Wizz Air", "Air Canada",
                     "British Airways", "Delta Air Lines", "Vueling Airlines", "TUIfly", "JetBlue Airways",
                     "Norwegian Air Shuttle", "Qantas", "Ryanair", "United Airlines", "Fly Dubai"]


# find cycles in graph
def cykle(Graf):
    H = G.to_directed()
    cycles = sorted(nx.simple_cycles(H))
    cycles.sort(key=lambda x: len(x))
    maxLength = max(len(x) for x in cycles)
    maxList = cycles[-1]
    if maxLength == 2:
        print("There are no cycles in graph other than direct")
    else:
        print(maxLength, maxList)


# calculate degree of graph
def stopnie_wierzcholkow(G):
    degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
    dmax = max(degree_sequence)


# calculating the most important airlines by number of unique routes (that are not covered by any other airline)
def najwaznejsze_linie_lotnicze():
    polaczenia = {}
    kto_obsluguje = {}
    iter = 0

    for flight in lista_polaczen:
        n1 = flight[0]  # name of first airport
        n2 = flight[2]  # name of second airport
        airline = [flight[1]]
        l1 = [n1, n2]
        l2 = [n2, n1]

        already_in = polaczenia.values()
        # checking if any other airline contains that route
        if l1 in already_in:
            index = list(polaczenia.values()).index(l1)

            temp = kto_obsluguje[index]
            temp.extend(airline)
            kto_obsluguje[index] = temp
        elif l2 in already_in:
            index = list(polaczenia.values()).index(l2)
            temp = kto_obsluguje[index]
            temp.extend(airline)
            kto_obsluguje[index] = temp
        else:
            polaczenia[iter] = l1
            kto_obsluguje[iter] = airline
            iter = iter + 1

    # calculate numbers of unique routes for each airline
    unikalne_polaczenia = []
    u_p = {}
    l_unikalnych_pol = []
    for key in list(kto_obsluguje.keys()):
        lista = kto_obsluguje[key]
        lista = list(dict.fromkeys(lista))

        if len(lista) == 1:
            a = lista[0]
            if a in list(u_p.keys()):
                temp = u_p[a]
                temp.append(polaczenia[key])
                u_p[a] = temp
            else:
                u_p[a] = [polaczenia[key]]

            if a in unikalne_polaczenia:
                id = unikalne_polaczenia.index(a)
                l_unikalnych_pol[id] = l_unikalnych_pol[id] + 1
            else:
                unikalne_polaczenia.append(a)
                l_unikalnych_pol.append(1)

    # check the max number of unique connections
    maksimum = max(l_unikalnych_pol) - 1

    for i in range(len(l_unikalnych_pol)):
        if l_unikalnych_pol[i] == maksimum:
            print(l_unikalnych_pol[i])
            print(airlines_dict[unikalne_polaczenia[i]][0])
    return polaczenia, kto_obsluguje, unikalne_polaczenia, u_p


# highest betweenness
def najwieksze_posrednictwo(Graph):
    posrednictwo = nx.edge_betweenness_centrality(Graph)
    krawedzie = list(posrednictwo.keys())
    centralna_krawedz = krawedzie[0]
    miara_posrednictwa = posrednictwo[centralna_krawedz]
    print(centralna_krawedz, miara_posrednictwa)


# calculate the most visited airport
def najwiecej_przelotow(Graph):
    polaczenia = {}
    ilosc_przelotow = {}
    iter = 0

    for flight in lista_polaczen:
        n1 = flight[0]  # name of first airport
        n2 = flight[2]  # name of second airport
        nr_of_flights = flight[3]
        l1 = [n1, n2]
        l2 = [n2, n1]

        already_in = polaczenia.values()

        if l1 in already_in:
            index = list(polaczenia.values()).index(l1)
            temp = ilosc_przelotow[index]
            temp = temp + nr_of_flights
            ilosc_przelotow[index] = temp
        elif l2 in already_in:
            index = list(polaczenia.values()).index(l2)
            temp = ilosc_przelotow[index]
            temp = temp + nr_of_flights
            ilosc_przelotow[index] = temp
        else:
            polaczenia[iter] = l1
            ilosc_przelotow[iter] = nr_of_flights
            iter = iter + 1
        # print(polaczenia)

    max_przelotow = 0
    for key in list(ilosc_przelotow.keys()):
        if ilosc_przelotow[key] > max_przelotow:
            max_przelotow = ilosc_przelotow[key]

    przeloty = []
    for key in list(ilosc_przelotow.keys()):
        if ilosc_przelotow[key] == max_przelotow:
            przeloty.append(polaczenia[key])

    print(max_przelotow, przeloty)


# find path between two nodes with DFS method
def DFS(visited, wierzcholki, krawedzie, node, sciezka):

    if not visited[node]:
        visited[node] = True
        sciezka.append(node)
        sasiedzi = []
        for para in krawedzie:
            if node == para[0]:
                sasiedzi.append(para[1])
            elif node == para[1]:
                sasiedzi.append(para[0])
        for neighbour in sasiedzi:
            DFS(visited, wierzcholki, krawedzie, neighbour, sciezka)
    return sciezka


# find the longest path
def najdluzsza_sciezka(Graph):
    edges = list(Graph.edges())

    paths = []
    for node_start in list(G.nodes()):
        visited = {}
        for i in list(Graph.nodes()):
            visited[i] = False
        paths.append(DFS(visited, nodes, edges, node_start, []))

    max_lenght = 0
    for p in paths:
        if len(p) > max_lenght:
            max_lenght = len(p)

    max_len_paths = []
    for p in paths:
        if len(p) == max_lenght:
            max_len_paths.append(p)

    print(max_lenght, max_len_paths)


# find paths between two nodes
def alternatywna_droga(Graph):
    polaczenia, kto_obsluguje, najwazniejsze_ll, unikalne_pol = najwaznejsze_linie_lotnicze()

    for arln in list(unikalne_pol.keys()):
        for pol in unikalne_pol[arln]:
            [n1, n2] = pol
            potencial_airlines = []
            for a in flight_by_airline.keys():
                tours = flight_by_airline[a]
                flag1 = False
                flag2 = False
                for temp in tours:
                    if n1 in temp:
                        flag1 = True
                    elif n2 in temp:
                        flag2 = True

                if flag1 and flag2:
                    if arln != a:
                        potencial_airlines.append(a)
            H = nx.Graph()      # create graph that doesn't contain path between two nodes
            new_shortest_path = ["a", "b", "c", "d", "e", "f"]
            arln_s_p = ''
            for b in potencial_airlines:
                list_of_flights = flight_by_airline[b]

                for flight in list_of_flights:
                    H.add_edge(flight[0], flight[2])

                if nx.has_path(H, n1, n2):
                    p = nx.shortest_path(H, source=n1, target=n2)
                    if len(p) < len(new_shortest_path):
                        new_shortest_path = p
                        arln_s_p = b
                H.clear()
            print("Połączenie ", pol, " realizowane przez ", arln, " może być zastąpione przez ", b)
            print("Aktualnie najkrótsza ścieżka, aby się tam dostać to: ", new_shortest_path)


# -------------------- Graph -----------------------
for flight in lista_polaczen:  # view all connections
    airline_name = airlines_dict[flight[1]]
    # print("From: ", airport_dict[flight[0]], " to: ", airport_dict[flight[2]], " with airline: ", airline_name[0])
    colour = airlines_dict[flight[1]]
    G.add_edge(airport_dict[flight[0]], airport_dict[flight[2]], weight=flight[3], color=colour[1])

print("Jakie bezpośrednie połączenie jest najczęściej wykorzystywane?")     # which direct connection is mostly used?
najwiecej_przelotow(G)
print("Które połączenie ma największą wartość pośrednictwa?")     # which connection has the highest betweenness?
najwieksze_posrednictwo(G)
print("Które połączenia to mosty?")     # which connection are bridges?
print(list(nx.bridges(G)))
print("Jaka jest możliwa najdłuższa ścieżka? Ile lotnisk odwiedza?")
# what is the longest possible path how many airports does it visit?
najdluzsza_sciezka(G)
print("Jakie linie lotnicze mają najwięcej połączeń, które nie są obsługiwane przez jakiekolwiek inne linie?")
# which airlines have the biggest number of connections, that are not operated by any other airlines?
najwaznejsze_linie_lotnicze()
print("Czy połączenia powyżej można obsłużyć jakoś inaczej?")
# does connections have alternative way?
alternatywna_droga(G)

edges = G.edges()
colors = [G[u][v]['color'] for u, v in edges]
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos=nx.circular_layout(G), edge_labels=labels)

nx.draw(G, edge_color=colors, with_labels=True, pos=nx.circular_layout(G))
plt.show()

# stopnie_wierzcholkow(G)
print("Średnia długość najkrótszych ścieżek", nx.average_shortest_path_length(G))    # average shortest paths length
G.clear()

airlines = flight_by_airline.keys()
average_shortest_path_per_airline = {}
average_shortest_path_per_low_cost_airline = {}

for arln in airlines:  # vie connections for each airline
    list_of_flights = flight_by_airline[arln]
    for flight in list_of_flights:
        colour = airlines_dict[arln]
        G.add_edge(airport_dict[flight[0]], airport_dict[flight[2]], weight=flight[3], color=colour[1])

    full_name = airlines_dict[arln]
    airline_name = full_name[0]

    edges = G.edges()
    colors = [G[u][v]['color'] for u, v in edges]
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos=nx.circular_layout(G), edge_labels=labels)

    print("Cykle tworzone przez samoloty linii ", airline_name)     # Cycles created by connections or airline
    cykle(G)
    try:
        average = nx.average_shortest_path_length(G)
        print("Średnia długość najkrótszych ścieżek dla linii wynosi ", average)    # average shortest paths length
        if airline_name in low_cost_airlines:
            average_shortest_path_per_low_cost_airline[arln] = average
        else:
            average_shortest_path_per_airline[arln] = average
    except:
        nx.NetworkXError("Graph is not connected.")
    else:
        print("Nie można obliczyć śderniej długości najkrótszych ścieżek")

    nx.draw(G, edge_color=colors, with_labels=True, pos=nx.circular_layout(G))
    plt.title(airline_name)
    plt.show()
    G.clear()

limit_average_lc_airline = 10
limit_average_airline = 10
for key in list(average_shortest_path_per_low_cost_airline.keys()):
    if average_shortest_path_per_low_cost_airline[key] < limit_average_lc_airline:
        limit_average_lc_airline = average_shortest_path_per_low_cost_airline[key]
for key in list(average_shortest_path_per_airline.keys()):
    if limit_average_airline > average_shortest_path_per_airline[key]:
        limit_average_airline = average_shortest_path_per_airline[key]

print(limit_average_airline, limit_average_lc_airline)

list_average = []
for key in list(average_shortest_path_per_airline.keys()):
    list_average.append(average_shortest_path_per_airline[key])
list_l_c = []
for key in list(average_shortest_path_per_low_cost_airline.keys()):
    list_l_c.append(average_shortest_path_per_low_cost_airline[key])

print(sum(list_average) / len(list_average), sum(list_l_c) / len(list_l_c))
print(statistics.variance(list_average), statistics.variance(list_l_c))
