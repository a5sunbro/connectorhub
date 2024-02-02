

function []=EEG_GAD(signal,matrix)
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
f_tilde_all = zeros(20,64,50);
anom_scores = zeros(20,64,50);
disp(['For patient number: ', int2str(patient)]);

names = [
            "FP1"
            "AF7"
            "AF3"
            "F1"
            "F3"
            "F5"
            "F7"
            "FT7"
            "FC5"
            "FC3"
            "FC1"
            "C1"
            "C3"
            "C5"
            "T7"
            "TP7"
            "CP5"
            "CP3"
            "CP1"
            "P1"
            "P3"
            "P5"
            "P7"
            "P9"
            "PO7"
            "PO3"
            "O1"
            "IZ"
            "OZ"
            "POZ"
            "PZ"
            "CPZ"
            "FPZ"
            "FP2"
            "AF8"
            "AF4"
            "AFZ"
            "FZ"
            "F2"
            "F4"
            "F6"
            "F8"
            "FT8"
            "FC6"
            "FC4"
            "FC2"
            "FCZ"
            "CZ"
            "C2"
            "C4"
            "C6"
            "T8"
            "TP8"
            "CP6"
            "CP4"
            "CP2"
            "P2"
            "P4"
            "P6"
            "P8"
            "P10"
            "PO8"
            "PO4"
            "O2"
        ];
f_intermediate = signal(patient);
A_intermediate = matrix(patient);
f_intermediate = f_intermediate{1};
A_intermediate = A_intermediate{1};
comm_num = numel(A_intermediate);   %Get the count of the elements in cell
end_comm_plus = 0;
for comm=1:comm_num
f = full(f_intermediate{comm});
A = full(A_intermediate{comm});
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
pred = pred + end_comm_plus;
if pred > 0
    disp(['For community number: ', int2str(comm)]);
    disp(names(pred));
end
% hubnodes(pred,patient) = 1;
% f_tilde_all(patient,:,:) = f_tilde;
% anom_scores(patient,:,:) = alternative_scores;
end_comm_plus = end_comm_plus + size(A,2);




end
end