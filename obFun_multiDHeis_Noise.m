function [ out,grad ] = objectiveFun_diag_reg_1_RTemp(S,J_fTemp)
global i D l m l2_r nd nCoun
J0=sparse(l,m,J_fTemp(1:nCoun-1),size(S,1),size(S,1));
J=full(J0);
J=J';
sigma=J_fTemp(nCoun);
if nargout >1
for n=1:nd;
A(:,:,n)=J*S(:,:,n)./sigma^2;
end
modA=sum(A.^2,3).^0.5;
temp_modA=modA;
[r]=find(temp_modA==0); 
temp_modA(r)=1e-8;
Zi=(2*pi)^(nd/2).*besseli((nd-2)/2,temp_modA)./(temp_modA).^((nd-2)/2);
%Zi=(2*pi)^1.5*(2^(-1/2)/(pi^0.5*gamma(1))).*2*sinh(temp_modA)./temp_modA;
temp2_modA=temp_modA;
[r]=find(temp2_modA>=700); 
temp2_modA(r)=700;
Cgrad=besseli((nd-2)/2+1,temp2_modA)./besseli((nd-2)/2,temp2_modA);
%Cgrad=(1./tanh(temp_modA)-1./temp_modA);
for n=1:nd;
temp_A = A(:,:,n);
[r]=find(temp_A==0); 
temp_A(r)=1e-8;
out_n(n) = -(1/size(S,2))*sum(sum((S(:,:,n).*temp_A-(1/nd).*log(Zi)),2));
grad0(:,:,n)=(1/size(S,2)).*(S(:,:,n)*S(:,:,n)'-(Cgrad.*temp_A./temp_modA)*S(:,:,n)')./sigma^2;
grad_sigma(n)=-(1/sigma)*2*(1/size(S,2)).*sum(sum(S(:,:,n).*temp_A-(Cgrad.*temp_A./temp_modA).*temp_A,2));
end
%out=sum(out_n); % no regularizer
%out=sum(out_n)+l2_r*sum(sum(abs(J))); % regularizer  l1
out=sum(out_n)+l2_r*(sum(sum(J.^2)))^0.5; % regularizer l2
%out=sum(out_n)+l2_r*(sum(sum(J.^4)))^0.25; % regularizer l4
grad0sum=sum(grad0,3);
grad_sigmasum=sum(grad_sigma);
[r]=find(grad0sum==0); 
grad0sum(r)=1e-8;
grad0=zeros(size(S,1),size(S,1));
grad0(i,:)=grad0sum(i,:);
grad0=grad0.*D;
[t,g,gradf]=find(grad0);
%grad=cat(1,-gradf,-grad_sigmasum); %no regularizer
%grad=-gradf; % no regularizer
%grad=-gradf+l2_r.*sign(J_f); % regularizer l1
grad=cat(1,-gradf+l2_r*J_fTemp(1:nCoun-1)./(sum(sum(J_fTemp(1:nCoun-1).^2)))^0.5,-grad_sigmasum); % regularizer l2
%grad=-gradf+l2_r*J_f./(sum(sum(J.^2)))^0.5; % regularizer l2
%grad=-gradf+l2_r*4.*J_f.^3./(sum(sum(J.^4)))^0.75; % regularizer l4
end
end
