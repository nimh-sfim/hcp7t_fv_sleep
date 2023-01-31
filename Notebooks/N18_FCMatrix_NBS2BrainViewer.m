%% Transform results so that they can be loaded in BrianNet
%  ========================================================
clear all
RESOURCES_DIR="/data/SFIMJGC_HCP7T/hcp7t_fv_sleep/Resources_NBS/";
scenarios=["Reference", "GSR", "BASIC", "BASICpp", "COMPCOR","COMPCORpp"];
contrasts=["AgtD","DgtA"];
for scenario = scenarios
    for contrast = contrasts
        input_path   = RESOURCES_DIR + "/NBS_ET_Results/NBS_ET_"+scenario+"_"+contrast+".mat";
        output_path  = RESOURCES_DIR + "/NBS_ET_Results/NBS_ET_"+scenario+"_"+contrast+".txt";
        output_path2 = RESOURCES_DIR + "/NBS_ET_Results/NBS_ET_"+scenario+"_"+contrast+".edge";
        if exist(input_path, 'file') == 2
            % File exists.
            data = load(input_path);
            data = full(cell2mat(data.nbs.NBS.con_mat));
            data = data + data.';
            writematrix(data,output_path,"Delimiter"," ");
            movefile(output_path,output_path2);
            disp("++ INFO: ["+scenario+","+contrast+"] --> Number of Significant Connections is " + (sum(sum(data))/2) + " originated in " + sum(sum(data)>0) + " nodes.")
        else
            % File does not exist.
            disp("Data does not exists for ["+scenario+","+contrast+"]");
        end
    end
end
