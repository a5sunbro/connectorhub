

function []=EEG_GAD(comm0,A0,f0)   %comm0 is in comm, A0 and f0 is in eeg_signal
%  Input: A0: Adjacency matrix NxN
%         f0: Graph signal matrix Nxp
%  Output: H: Filter response
%  Author: Meiby Ortiz-Bouza
%  Address: Michigan State University, ECE
%  email: ortizbou@msCu.edu

%%% Parameters
for patient=1:20
T=3;  
rho=1;
alpha=0.5;
hubnodes = zeros(64,20);

disp(['For patient number: ', int2str(patient)]);
names = readlines('/home/duc/connectorhub/glasser360-master/glasser360NodeNames.txt','EmptyLineRule','skip');


f_intermediate = transpose(squeeze(f0(patient,:,:)));
%f_intermediate = squeeze(f0(patient,:,:));
A_intermediate = squeeze(A0(patient,:,:));

comm_intermediate = comm0(patient);
comm_intermediate = comm_intermediate{1};
comm_num = numel(comm_intermediate);   %Get the count of the elements in cell

% comm_index = comm_intermediate{sub_comm} + 1; %Every node in python is lower than MATLAB by 1 due to indices standard
% 
% f = f_intermediate(comm_index,:);   %for the signal, we only need to get signal from indices
% G = graph(A_intermediate, names);    %ID for each node
% sub_G = subgraph(G, comm_index);          %ID is retained after using subgraph
% 
% A = full(adjacency(sub_G, 'weighted'));
G = graph(A_intermediate, names);

f = f_intermediate;
A = A_intermediate;
[~,N]=size(A);
[~,p]=size(f);  



%% Learn filter
An = normadj(A);   % normalized Adjacency
Ln = eye(N)-An;   % normalized Laplacian
[U,d]=eig(full(Ln));
D=diag(d);

%%%  t-th shifted input signal as S(t) := U'*D^t*U'*F
for t=1:T
zt{t}=U*d^(t-1)*U'*f;
end

for i=1:N
    for t=1:T
    zn(t,:,i)=zt{t}(i,:);
    end
end

ZN1 = permute(sum(pagemtimes(permute(zn, [2 3 1]), pagemtimes(Ln,permute(zn, [3 2 1]))), [1 2]), [3 1 2]);

%% Initializations
mu1=rand(N,p);
V=mu1/rho;
h=rand(T,1);
h=h/norm(h);
H=0;
for t=1:T
    Hnew=H+h(t)*diag(D.^(t-1));
    H=Hnew;  
end

thr=alpha/rho;
for n=1:40
    %% ADMM (Z,h,V)
    %%% B^(k+1) update using h^k and V^k
    X=(eye(N)-U*H*U')*f-V;
    B=wthresh(X,'s',thr);
    %%% h^(k+1) update using B^(k+1) and V^k
    E=B-f+V;
    count1=0;
    count2=0;
    SZ=0;
    for t=1:p
    for k=1:N
        count1=0;
        SZnew=SZ+sum(ZN1,3);
        SZ=SZnew;
        count2=count2+1;
        ZN2(:,:,count2)=zn(:,t,k)*zn(:,t,k)';
        b(:,:,count2)=zn(:,t,k)*E(k,t);
    end
    end
    Y=2*SZ+rho*sum(ZN2,3);
    h_new=-inv(Y)*rho*sum(b,3);
    h_new=h_new/norm(h_new);

    H=0; %% C filter for next iteration
    for t=1:T
        Hnew=H+h_new(t)*diag(D.^(t-1));
        H=Hnew;  
    end

    %%% V^(k+1) update using V^k, Z^(k+1), and c^(k+1)
    V_new=V+rho*(B-(eye(N)-U*H*U')*f);
    if norm(h_new-h)<10^-3
        break
    end
    h=h_new;
    V=V_new;
end
clear b ZN2 ZN1 zn zt


f_tilde=U*H*U'*f;

% % %%% Anomaly scoring based on smoothness
for i=1:N
     s=A(i,:).*((f(i,:) - f).^2)';
     e0(i,:)=sum(s, 2);
end
for i=1:N
     s=A(i,:).*((f_tilde(i,:) - f_tilde).^2)';
     en(i,:)=sum(s, 2);
end
alternative_scores = e0 - en;
clear e0
clear en
clear s

for i=1:N
     s=A(i,:).*vecnorm((f - f(i,:))');
     e0(i)=sum(s);
end

for i=1:N
     s=A(i,:).*vecnorm((f_tilde - f_tilde(i,:))');
     en(i)=sum(s);
end

scores=e0-en;

clear e0
clear en
clear s


alternative_scores = zscore(alternative_scores, 1, 1 );
scores = zscore(scores, 1, 'all');
 % % 

threshold = 3;

pred = find(scores > threshold| scores < -threshold);
% pred_name = string(sub_G.Nodes{pred,:});
pred_name = string(G.Nodes{pred,:});
if pred > 0
    for n=1:length(pred_name)   % for every hub node detected
        all_connectivity = centrality(G, 'degree', 'Importance', G.Edges.Weight);
        all_connectivity = all_connectivity(findnode(G, pred_name(n)));
        participation_index = 1;
        outer_connectivity = 0;
        % this part is for calculating the participation index P
        check = 0;
        for i=1:comm_num
            k = findnode(G, pred_name(n));
            if ismember(k,(comm_intermediate{i} + 1))==1
                sub_comm = i;
            end
        end
        for sub_sub_comm=1:comm_num
            %This is for finding the community the hub node is in

            if sub_sub_comm ~= sub_comm     %get the subgraph of every other community with the targeted node
                comm_index_other = comm_intermediate{sub_sub_comm} + 1; %Every node in python is lower than MATLAB by 1 due to indices standard
                comm_index_other(end + 1) = k;   %Adding the predicted node onto the community
                sub_G = subgraph(G, comm_index_other);          %ID is retained after using subgraph
            else
                comm_index_other = comm_intermediate{sub_sub_comm} + 1;
                sub_G = subgraph(G, comm_index_other);
            end

            deg_ranks = centrality(sub_G, 'degree', 'Importance', sub_G.Edges.Weight);   %finding the degree of the targeted node
            outer_connectivity = deg_ranks(findnode(sub_G, pred_name(n)));
            check = check + outer_connectivity;
            participation_index = participation_index - (outer_connectivity/all_connectivity)^2;

        end
        
        disp(['For community number: ', int2str(sub_comm)]);
        disp(pred_name(n));
        disp(participation_index);

    end
end

% hubnodes(pred,patient) = 1;
% f_tilde_all(patient,:,:) = f_tilde;
% anom_scores(patient,:,:) = alternative_scores;




end