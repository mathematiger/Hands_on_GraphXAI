# Here, several functions for visualization & their utils are implemented



from ce_generation import generate_cedict_from_ce
import dgl
import networkx as nx
import colorsys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import re
import os

from graph_generation import graphdict_and_features_to_heterodata




# ---------------- utils
def uniquify(path, extension = '.pdf'):
    if path.endswith("_"):
        path += '1'
        counter = 1
    while os.path.exists(path+extension):
        counter +=1
        while path and path[-1].isdigit():
            path = path[:-1]
        path +=str(counter)
    return path


def remove_integers_at_end(string):
    pattern = r'\d+$'  # Matches one or more digits at the end of the string
    result = re.sub(pattern, '', string)
    return result


def get_last_number(string):
    pattern = r'\d+$'  # Matches one or more digits at the end of the string
    match = re.search(pattern, string)
    if match:
        last_number = match.group()
        return int(last_number)
    else:
        return None


def generate_colors(num_colors):
    # Define the number of distinct hues to use
    num_hues = num_colors + 1
    # Generate a list of evenly spaced hues
    hues = [i / num_hues for i in range(num_hues)]
    # Shuffle the hues randomly
    #random.shuffle(hues)
    saturations = []
    #saturations = [0.8 for _ in range(num_colors)]
    values = []
    #values = [0.4 for _ in range(num_colors)]
    for i in range(num_colors):
        if i % 2 == 0:
            values.append(0.4)
            saturations.append(0.4)
        else:
            values.append(0.8)
            saturations.append(0.7)
    # Convert the hues, saturations, and values to RGB colors
    colors = [colorsys.hsv_to_rgb(h, s, v) for h, s, v in zip(hues, saturations, values)]
    # Convert the RGB colors to hexadecimal strings
    hex_colors = [f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}" for r, g, b in colors]
    return hex_colors





# ----------------- Visualization Functions

# old function, but a lot simpler then visualize_heterodata
def visualize(dict_graph):
    graph_final = dgl.heterograph(dict_graph)
    graph_final_hom = dgl.to_homogeneous(graph_final)
    graph_final_nx = dgl.to_networkx(graph_final_hom)
    options = {
        'node_color': 'black',
        'node_size': 20,
        'width': 1,
    }
    plt.figure(figsize=[15, 7])
    nx.draw(graph_final_nx,**options)
    name_plot_save = 'content/plots/graph.pdf'
    name_plot_save = uniquify(name_plot_save)
    #plt.figure()
    plt.savefig(name_plot_save, format="pdf")
    plt.show()
    plt.clf()


# This function is the main function to visualize
def visualize_heterodata(hd, addname = '', ce = None, gnnout = None, mean_acc = None, add_out = None, list_all_nodetypes = None, label_to_explain = None, name_folder = ''):
    try:
        plt.clf()
    except Exception as e:
        print('plt could not be clarified')
    options = {
        'with_labels' : 'True',
        'node_size' : 500
    }
        # create random colours for visualization
    number_of_node_types = len(hd.node_types)
    number_of_node_types_for_colors = number_of_node_types
    curent_nodetypes_to_all_nodetypes = []
    for _ in range(len(hd.node_types)):
        if list_all_nodetypes != None:
            all_nodetypes_index = list_all_nodetypes.index(hd.node_types[_])
        else:
            all_nodetypes_index = _
        curent_nodetypes_to_all_nodetypes.append([_, all_nodetypes_index])
    if list_all_nodetypes != None:
        number_of_node_types_for_colors = len(list_all_nodetypes) 
    colors = generate_colors(number_of_node_types_for_colors)
    if number_of_node_types_for_colors == 4:
        colors = ['#59a14f', '#f28e2b', '#4e79a7', '#e15759']
    #find out, which node in homogeneous graph has which type
    homdata = hd.to_homogeneous()
    tensor_with_node_types = homdata.node_type
    #generate networkx graph with the according setting
    Gnew = nx.Graph()
    #add nodes
    num_nodes_of_graph = len(homdata.node_type.tolist())
    Gnew.add_nodes_from(list(range(num_nodes_of_graph)))
    #add edges
    list_edges_start, list_edges_end = homdata.edge_index.tolist()[0], homdata.edge_index.tolist()[1]
    list_edges_for_networkx = list(zip(list_edges_start, list_edges_end))
    Gnew.add_edges_from(list_edges_for_networkx)
    #color nodes
    list_node_types = homdata.node_type.tolist()
    node_labels_to_indices = dict()
    index = 0
    stop = False #the prediction is always done for the first node
    for nodekey in homdata.node_type.tolist():
        if label_to_explain != None:
            if str(curent_nodetypes_to_all_nodetypes[nodekey][1]) == label_to_explain and stop == False:
                node_labels_to_indices.update({index : '*'})
                stop = True
            else:
                node_labels_to_indices.update({index : ''})
        else:
            node_labels_to_indices.update({index : curent_nodetypes_to_all_nodetypes[nodekey][1]})
        index +=1
    color_map_of_nodes = []
    for typeindex in list_node_types:
        color_map_of_nodes.append(colors[curent_nodetypes_to_all_nodetypes[typeindex][1]])
    #plt
    nx.draw(Gnew, node_color=color_map_of_nodes,  **options, labels = node_labels_to_indices)
    #create legend
    patch_list = []
    name_list = []
    for i in range(number_of_node_types):
        patch_list.append(plt.Circle((0, 0), 0.1, fc=colors[curent_nodetypes_to_all_nodetypes[i][1]]))
        name_list.append(hd.node_types[i])
    
    #create caption
    caption_text = ''
    caption_size = 18
    if ce != None:
        caption_text += ce
        caption_size -= 4
    if gnnout != None:
        caption_text += '\n'+ ' out: ' + str(gnnout)
        caption_size -= 4
    if mean_acc != None:
        caption_text += ' acc: ' + str(mean_acc)
        caption_size -= 4
    if add_out != None:
        caption_text += ' ' + add_out
        caption_size -= 4   
    caption_position = (0.5, 0.1)  # Adjust the position as per your requirements
    
    
    #folder to save in:
    folder = remove_integers_at_end(addname)
    number_ce = get_last_number(addname)
    print(folder, number_ce)
    #goal: ce_name_1_graph_3
    if name_folder == '':
        name_plot_save = 'content/plots/'+folder+'/ce_'+folder+'_'+str(number_ce)+'_graph_'
    else:
        name_plot_save = name_folder + '/ce_'+str(number_ce)+'_graph_'
    name_plot_save = uniquify(name_plot_save, '_wo_text.pdf')
    #name_plot_save = name_plot_save.replace(".pdf", "")
    #name_plot_save += 'wo_text.pdf'
    plt.savefig(name_plot_save+'_wo_text.pdf', bbox_inches='tight', format="pdf")
    plt.legend(patch_list, name_list)
    plt.figtext(*caption_position, caption_text, ha='center')#, size = caption_size)
    #plt.figure()
    #name_plot_save = 'content/plots/'+folder+'/'+'graph'+addname+'.pdf'
    name_plot_save = uniquify(name_plot_save)
    plt.savefig(name_plot_save+'.pdf', bbox_inches='tight', format="pdf")
    #plt.show()


def visualize_best_results(num_top, saved_list, addname = '', list_all_nodetypes = None, label_to_explain = None):
    num_top = min([num_top, len(saved_list)])
    for i in range(num_top):
        saved_dict_result = saved_list[i][0][0]
        saved_features_result = saved_list[i][0][1]
        ce = None
        gnn_out = None
        mean_acc = None
        try:
            gnn_out = saved_list[i][1]
        except Exception as e:
            print(f"290 Here we skiped the error: {e}")
        try:
            ce = saved_list[i][2]
        except Exception as e:
            print(f"290 Here we skiped the error: {e}")
        try:
            mean_acc = saved_list[i][3]
        except Exception as e:
            print(f"290 Here we skiped the error: {e}")
        visualize_heterodata(graphdict_and_features_to_heterodata(saved_dict_result, saved_features_result), addname, gnnout = gnn_out, ce = ce, mean_acc = mean_acc, list_all_nodetypes = list_all_nodetypes, label_to_explain = label_to_explain)
        '''
        try:
            visualize_heterodata(graphdict_and_features_to_heterodata(saved_dict_result, saved_features_result), addname, gnnout = gnn_out, ce = ce, int_generate_colors = int_generate_colors)
        except Exception as e:
            print(f"290 Here we skiped the error: {e}")
            print(291, i, saved_list[i])
           ''' 

def visualize_best_ces(num_top_ces, num_top_graphs, list_of_results_ce = list(), list_all_nodetypes = None, label_to_explain = None, ml = None, ds = None, node_expl = None, plotname = 'any', random_seed = 615):
    #for each CE: visualize num_top_graphs by saving them under a unique name
    #aufbau: [graphs, ce]
    num_top_ces = min(num_top_ces, len(list_of_results_ce))
    for ind_ce in range(num_top_ces):
        if ml != None and ds != None and node_expl != None:
            ce_fid = ce_fidelity(list_of_results_ce[ind_ce][5], modelfid=ml, datasetfid = ds, node_type_expl = node_expl, random_seed = random_seed)
            print(378, ce_fid)
        else:
            ce_fid = -1
        ce_vis = list_of_results_ce[ind_ce][1]
        score = list_of_results_ce[ind_ce][2]
        avg_acc = list_of_results_ce[ind_ce][3]
        max_acc = list_of_results_ce[ind_ce][4]
        max_f = list_of_results_ce[ind_ce][5]
        # new way to compute accuracy directly on CE:
        ce_dict = generate_cedict_from_ce(list_of_results_ce[ind_ce][6])
        #compute_confusion_for_ce_line(ce_dict)
        
        
        print(364, avg_acc, ce_vis, num_top_ces)
        graphs = list_of_results_ce[ind_ce][0]
        addname = plotname+str(ind_ce+1)
        num_top_graphs_local = min(num_top_graphs, len(graphs))
        for ind_gr in range(num_top_graphs_local):
            dict_graph_ce = graphs[ind_gr][0][0]
            features_graph_ce = graphs[ind_gr][0][1]
            gnn_out = None
            try:
                gnn_out = graphs[ind_gr][1]
            except Exception as e:
                print(f"290 Here we skiped the error: {e}")
            visualize_heterodata(graphdict_and_features_to_heterodata(dict_graph_ce, features_graph_ce), addname, gnnout = 'Score: '+str(score), ce = ce_vis, mean_acc = str(avg_acc)+' max acc: ' + str(max_acc) + ' F1: ' + str(max_f)+' fid: ' + str(ce_fid), list_all_nodetypes = list_all_nodetypes, label_to_explain = label_to_explain)





def visualize_one_hd(ce, path, number_of_graphs, add_result):
    print('Visualization started')
    # visualize graphs

    


            
            
            
            
            
            
            
            
            
            
            
            
            
            