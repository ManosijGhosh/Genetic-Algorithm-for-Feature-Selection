%Initial or First Level Feature Selection (MA 1st level)
%the network, population stored in results.mat, copy into Run folder if better accuracy achieved
function []=main()
    tic
    rng('shuffle');
    %{
    x=load('Data2/train.mat');   % Training File 
    x=x.train;
    t=load('Data2/totalTrainLabel.mat');  % Training label
    t=t.totalTrainLabel;
    x2=load('Data2/test.mat');   % Training Test File
    x2=x2.test;
    t2=load('Data2/totalTestLabel.mat');   % Training Test Label
    t2=t2.totalTestLabel;
    disp('Imports done');
    %}
    x=importdata('data/Input.xlsx');
    t=importdata('data/target.xlsx');
    %size(t)
    ftranks=importdata('data/franks.txt');
    chr=importdata('data/selection.xlsx');
    
    x2=x(chr(:)==1,ftranks(1:200));
    t2=t(chr(:)==1,:);
    x=x(chr(:)==0,ftranks(1:200));
    t=t(chr(:)==0,:);
    disp('imports done');
    [~,c]=size(x);
    %n=int16(input('Enter the number of chromosomes to work on :'));
    n=10 ;   % To change population Size
    %mcross=int16(input('Enter the maximum number of crossovers to do :'));
    mcross=int16(5);
    size(x)
    population=datacreate(n,c);   % Feature Length 
    fprintf('data created\n');
    %[r,c]=size(population);
    rank=zeros(1,n);
    rankcs=zeros(1,n);
    netArray=cell(n,1);
    
    [population,rank,netArray]=chromosomeRank(x,t,x2,t2,population,rank,netArray,0);
    fprintf('Chromosomes ranked\n');
    
    %{
        [r,c]=size(t2);
        target=t2(1:r,1:c);
        input=x2(1:r,population(1,:)==1);
        
        inputs = input';
        targets = target';
        outputs = netArray{1}(inputs); %this is how saved net is to be used later

        [c, ] = confusion(targets,outputs);
        fprintf('Percentage Correct Classification   : %f%%\n', 100*(1-c));
        %}
    str=strcat('ResultStore/result','.mat');
    fnum=25;acc=0.99;count=int16(1);%fum- length of reduced feature set desired; acc- accuracy of reduced set desired
    %frank gives position where the features are whose rank is index
    %ftrank(1)=position of feature of rank 1.
    while ((sum(population(1,:)==1)>fnum || rank(1)<acc) && (count<=30))    % To Change if reqd..count - number of iterations
        
        
        %crossover starts
        fprintf('\nCrossover done for %d th time\n',count);
        limit = randi(mcross-2,1)+2;%assmming at max m crossovers
        %(mod(rand(1,int16),(n))+1)
        for i=1:limit
            %cumulative sum for crossover
            rankcs(1:n)=rank(1:n);%copying the values of rank to rankcs
            for j= 2:n% size of weights = no. of features in popaulation=c
                rankcs(j)=rankcs(j)+rankcs(j-1);
            end
            maxcs=rankcs(n);
            for j= 1:n
                rankcs(j)=rankcs(j)/maxcs;
            end            
            a=find(rankcs>rand(1),1,'first');
            b=find(rankcs>rand(1),1,'first');
            %roulette wheel ends
            
            [population,rank,netArray]=crossover(x,t,x2,t2,population,a,b,rand(1),rand(1),rank,netArray);
            %[population,rank]=crossover(x,t,population,randi(n,1),randi(n,1),rand(1),rank);
            clear a b j rankcs;
        end
        %crossover ends
        
        count=count+1;
        [population,rank,netArray]=chromosomeRank(x,t,x2,t2,population,rank,netArray,1);
        
        %if ( count==5 || count==10 || count == 15 )
            disp('Results saved');
            save(str,'population','rank','netArray');
        %end
    end
    fprintf('The least number of features is : %d\n',sum(population(1,:)==1));
    fprintf('The best accuracy is : %d\n',rank(1));
    save(str,'population','rank','netArray');
    disp('Final results stored');
    %{
    %error and net check
    firstL=matfile('results.mat');
    nets=firstL.netArray;
    for i=1:n
        net=nets{i};
        view(net);
        out=net(x2(population(i,:)==1));
        [c, ]=confusion(t2',out);
        fprintf('The size of %d = %d and accuracy is %f\n',i,sum(population(i,:)==1),((1-c)*100));
    end
    %}
    toc
end