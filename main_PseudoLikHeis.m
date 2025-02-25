% This script finds the values of the elements of the matrix J that maximize
% the single-coutry pseudolikelihood function for each country. The
% hamiltonian is written as a function of the scalar product of the disciplinary profile of a couple of countries.
% The temperature is a free fitting parameter, see sigma parameter in the following.
clear all
close all
%
global i D l m l2_r nd nCoun
%
filedati_title='ESI Mod1 post92 50c reg norma1 '; %choose a name for the output files
ndisc=14; % dimension of the spin variables (total number of disciplines)
d=(1:14); % vector labeling the d-th component of the spin variables;
for j=1:nd;
datafiles=['Input Files Name_',num2str(d(j))]; % a single input file is available for each component of the spin variables. The input files %corresponding to the d-th component of the spin variables are named 'Input Files Name_d', where 'd' is a number labeling the dimensions of the %spin variables. The input files of each component contain a time series of graph's nodes configurations.;
data=importdata(datafiles);
S(:,:,j) = data; % matrix containing the time series of spin variables. First index of the matrix: node's index. Second index of the matrix: %time or configuration index. Third index of the matrix: component index of the spin variable.
end
ci=[1];% array containing the initial values of the elements of the matrix J in the maximization algorithm. Do not put value ==0! Zero values %of elements of J are used to disentagle fixed parameters and free parameters in the optimization routine.
l2=[0.13]; %  array containing the values of the parameer of the regularizer of the Log-Pseudo_likelihood function one wants to try.
%
sigma_singC=zeros(1,nCoun);
for r=1:length(l2);
    l2_r=l2(r);
for t=1:length(ci);
for i=1:nCoun; % the pseudolikelihood is maximized for single country
sigma=1; % initial value of $\sigma^2$=K_B*T
Jinit = zeros(size(S,1),size(S,1));
D=1-diag(diag(ones(size(S,1),size(S,1))));
Jinit(:,i)=Jinit(:,i)+ci(t); % inizialization of the elelements of the matrix J
Jinit=Jinit.*D;
[l,m,jf]=find(Jinit);
Jinit_f=jf;
Jinit_fTemp=cat(1,Jinit_f,sigma);
options=optimoptions('fminunc','Algorithm','trust-region','Display','iter',...
     'GradObj','on','MaxFunEvals',100000,'TolX',1e-6,'TolFun',1e-6,'MaxIter',1000);
%options = optimoptions('fmincon','SpecifyObjectiveGradient',true,'Display','iter','MaxFunEvals',100000,'TolX',1e-12,'TolFun',1e-9);
lb_J=-abs(Jinit_fTemp(1:nCoun-1).*inf);
ub_J=abs(Jinit_fTemp(1:nCoun-1).*inf);
lb_sigma=Jinit_fTemp(nCoun).*0.01;
ub_sigma=Jinit_fTemp(nCoun).*inf;
lb=cat(1,lb_J,lb_sigma);
ub=cat(1,ub_J,ub_sigma);
A = [];
b = [];
Aeq = [];
beq = [];
nonlcon = [];
objfun = @(J_fTemp) obFun_multiDHeis_Noise(S,J_fTemp); % it is defined the function to be minimized, objfun, given in obFun_multiDHeis_Noise.m.
[J_fTemp,e,exitflag,output,grad] = fminunc(objfun,Jinit_fTemp,options);% Minimization algorithm by fminunc [https://www.mathworks.com/help optim/ug/fminunc.html]. The gradient is included. The upper and lower bounds of J are not considered here.
%[J_fTemp,e,exitflag,output,grad]=fmincon(objfun,Jinit_fTemp,A,b,Aeq,beq,lb,ub,nonlcon,options); %here the upper and lower bounds of J are considered. %Uncomment this line if you want to include upper and lower bounds of J.
J0=sparse(l,m,J_fTemp(1:nCoun-1),size(S,1),size(S,1));
J_t=full(J0);
J(:,i)=J_t(:,i);
sigma_singC(i)=J_fTemp(nCoun);
end
J_ciR(:,:,t,r)=triu((J+J')./2); % the J matrix is made symmetric
%J_ci=J(:,:,t,r); % uncomment this line if you want to save the J matrix forall ci
%save (['J_',output_name,'ci=',num2str(ci(t)),'l2=',num2str(l2(r))], 'J_ci', '-ascii'); % uncomment this line if you want to save the J matrix forall ci
if t==1;
    s=t;
else
    s=t-1;
end
if  J_ciR(:,:,t,r)-J_ciR(:,:,s,r) ~= 0;  
    disp('DEPENDENCE FROM INITIAL CONDITIONS');
    figure1=figure;
    % Create axes
    axes1 = axes('Parent',figure1);
    imagesc(J_ciR(:,:,t,r));
    title(['J_',filedati_title,' ci=',num2str(ci(t)),' l2=',num2str(l2(r))]);
    % Create colorbar
    colorbar('peer',axes1);
    break %the script is stopped when a dependence from the initial condition of J is found
end
end
J_end=J_ciR(:,:,length(ci),r);
% save (['J_',output_name,'l2=',num2str(l2(r))], 'J_end', '-ascii'); % uncomment this line if you want to save the J matrix forall l2
figure1=figure;
axes1 = axes('Parent',figure1);
imagesc((J_end));
title(['J_',output_name,' l2=',num2str(l2(r))]);
% Create colorbar
colorbar('peer',axes1);
% saveas(figure,['J_',output_name,' l2=',num2str(l2(r)),'.fig']);
% saveas(figure,['J_',output_name,' l2=',num2str(l2(r)),'.jpg']);
%
figure1=figure;
% Create axes
axes1 = axes('Parent',figure1);
imagesc(sign(J_end));
title(['sign(J)_',output_name,' l2=',num2str(l2(r))]);
% Create colorbar
colorbar('peer',axes1);
% saveas(figure,['sign(J)_',output_name,' l2=',num2str(l2(r)),'.fig']);
% saveas(figure,['sign(J)_',output_name,' l2=',num2str(l2(r)),'.jpg']);
figure1=figure;
% Create axes
axes1 = axes('Parent',figure1);
imagesc(log10(abs(J_end)));
title(['log10(abs(J))_',output_name,' l2=',num2str(l2(r))]);
% Create colorbar
colorbar('peer',axes1);
% saveas(figure,['log10(abs(J))_',output_name,' l2=',num2str(l2(r)),'.fig']);
% saveas(figure,['log10(abs(J))_',output_name,' l2=',num2str(l2(r)),'.jpg']);
end
