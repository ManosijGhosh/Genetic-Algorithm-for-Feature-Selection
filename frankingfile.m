function []=frankingfile()
    x=importdata('dataUsedCurrent/Input.xlsx');
    tar=importdata('dataUsedCurrent/targetsF.xlsx');
    k=10;tar=tar';
    [ftrank,weights] = relieff(x,tar,k,'method','classification');%feature ranking
    bar(weights);
    xlabel('Predictor rank');
    ylabel('Predictor importance weight');
    clear tar k;
    fp=fopen('dataUsedCurrent/franks.txt','w');
    [~,c]=size(x);
    for i=1:c
        fprintf(fp,'%d\t',ftrank(i));
        fprintf('%d\t',ftrank(i));
    end
    fclose(fp);
    fprintf('\n');
end
    