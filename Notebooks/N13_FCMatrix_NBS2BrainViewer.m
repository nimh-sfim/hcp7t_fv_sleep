%% Transform results so that they can be loaded in BrianNet
%  ========================================================
clear all
RESOURCES_DIR="/data/SFIMJGC_HCP7T/hcp7t_fv_sleep/Resources/";
scenarios=["BASIC", "Behzadi_COMPCOR","AFNI_COMPCOR","AFNI_COMPCORp"];
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
            disp("++ INFO: ["+scenario+","+contrast+"] --> Number of Significant Connections is " + (sum(sum(data))/2))
        else
            % File does not exist.
            disp("Data does not exists for ["+scenario+","+contrast+"]");
        end
    end
end

%%        
%    result_AgtD = load("/data/SFIMJGC_HCP7T/Sleep/Resources/NBS_Results/NBS_ET_"+scenario+"_AgtD.mat");
%    result_DgtA = load("/data/SFIMJGC_HCP7T/Sleep/Resources/NBS_Results/NBS_ET_"+scenario+"_DgtA.mat");
%    result_AgtD = full(cell2mat(result_AgtD.nbs.NBS.con_mat));
%    result_DgtA = full(cell2mat(result_DgtA.nbs.NBS.con_mat));
%    result_AgtD = result_AgtD + result_AgtD.';
%    result_DgtA = result_DgtA + result_DgtA.';
%    writematrix(result_AgtD, ...
%                "/data/SFIMJGC_HCP7T/Sleep/Resources/NBS_Results_T4/NBS_ET_"+scenario+"_AgtD.txt", ...
%                "Delimiter"," ");
%    writematrix(result_DgtA, ...
%                "/data/SFIMJGC_HCP7T/Sleep/Resources/NBS_Results_T4/NBS_ET_"+scenario+"_DgtA.txt", ...
%                "Delimiter"," ");