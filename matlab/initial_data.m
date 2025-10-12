%% initial data
n_candidate=610;     
n_community=27;

Q = [30 40 50];
C=[45 65 80];
alpha=1;
beta=1;


U=25;

Cp= 20;
Cpp=50;
D=200;
LB=50;
UB=100;


landa=0.5;

% b=100; a=-100;
% xy_candidate=(b-a).*rand(n_candidate,2) + a;
% b=100; a=-100;
% xy_community=(b-a).*rand(n_community,2) + a;
% b=100; a=-100;
% xy_hospital=(b-a).*rand(n_hospital,2) + a;


% determining number of community demand
% a=10; b=20;
% Ep=floor((b-a).*rand(n_community,1)) + a;
% 
% a=30; b=40;
% Em=floor((b-a).*rand(n_community,1)) + a;
% 
% a=40; b=50;
% Eo=floor((b-a).*rand(n_community,1)) + a;

%% Coordination data
coordinate_data
%% Demand data
demand_data
Ep=.6*Em;
Eo=1.4*Em;
%%
E_L=((landa/2)*(Em+Eo)/2+(1-landa/2)*(Ep+Em)/2);
E_U=((1-landa/2)*(Em+Eo)/2+(landa/2)*(Ep+Em)/2);

% determining number of hospital demand
% a=20; b=40;
% Epp=floor((b-a).*rand(n_hospital,1)) + a;
% 
% a=40; b=50;
% Epm=floor((b-a).*rand(n_hospital,1)) + a;
% 
% a=50; b=70;
% Epo=floor((b-a).*rand(n_hospital,1)) + a;
% 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d=zeros(n_candidate,n_community);
dpp=zeros(n_candidate,n_candidate);

gama=zeros(n_candidate,n_community);

for i=1:n_candidate
    for j=1:n_community
        d(i,j)=sqrt((xy_community(j,2)-xy_candidate(i,2))^2+(xy_community(j,1)-xy_candidate(i,1))^2);
        if d(i,j)<=LB
            gama(i,j)=0;
        elseif d(i,j)>UB
            gama(i,j)=1;
        else
            gama(i,j)=(d(i,j)-LB)/(UB-LB);
        end
    end
end

for i=1:n_candidate
    for l=1:n_candidate
        dpp(i,l)=sqrt((xy_candidate(l,2)-xy_candidate(i,2))^2+(xy_candidate(l,1)-xy_candidate(i,1))^2);
    end
end