function [KE,iK,jK,F,U,fixeddofs,alldofs,freedofs,edofMat]=Q1(nu,nelx,nely,Gen_data)

A11 = [12  3 -6 -3;  3 12  3  0; -6  3 12 -3; -3  0 -3 12];
A12 = [-6 -3  0  3; -3 -6 -3 -6;  0 -3 -6  3;  3 -6  3 -6];
B11 = [-4  3 -2  9;  3 -4 -9  4; -2 -9 -4 -3;  9  4 -3 -4];
B12 = [ 2 -3  4 -9; -3  2  9 -2;  4  9  2  3; -9 -2  3  2];
KE = 1/(1-nu^2)/24*([A11 A12;A12' A11]+nu*[B11 B12;B12' B11]);
nodenrs = reshape(1:(1+nelx)*(1+nely),1+nely,1+nelx);
edofVec = reshape(2*nodenrs(1:end-1,1:end-1)+1,nelx*nely,1);
edofMat = repmat(edofVec,1,8)+repmat([0 1 2*nely+[2 3 0 1] -2 -1],nelx*nely,1);
iK = reshape(kron(edofMat,ones(8,1))',64*nelx*nely,1);
jK = reshape(kron(edofMat,ones(1,8))',64*nelx*nely,1);

if Gen_data==1
%% EX 1 : Column with 3 loads 
    indx = [2,2*(nely+1)*(nelx/4)+2,2*(nely+1)*(nelx/2)+2,2*(nely+1)*(3*nelx/4)+2,2*(nely+1)*(nelx)+2];
    F = sparse(indx,1,-1,2*(nely+1)*(nelx+1),1);
    %% Fixeddofs all bottom
    fixeddofs = [2*(nely+1)*(1/4*nelx)-1:2*(nely+1):2*(3/4*nelx+1)*(nely+1),2*(nely+1)*(1/4*nelx):2*(nely+1):2*(3/4*nelx+1)*(nely+1)];
    %fixeddofs = [2*(nely+1)-1:2*(nely+1):2*(nelx+1)*(nely+1),2*(nely+1):2*(nely+1):2*(nelx+1)*(nely+1)];
end
%% EX 2 : CANTILEVER BEAM with load
if Gen_data ==2
    % Load node 
    F = sparse(2*(nely+1)*(nelx+1),1,-1,2*(nely+1)*(nelx+1),1);
    % Fixeddofs all left
    fixeddofs = [1:2*nely+1];
end
%% EX 1 : L SHAPED
% Load bottom node
if Gen_data ==3
    % F = sparse(2*(nely+1)*(nelx)+2*nely,1,-1,2*(nely+1)*(nelx+1),1);
     F = sparse(2*(nely+1)*(nelx+1),1,-1,2*(nely+1)*(nelx+1),1);
    % Fixeddofs all top
    fixeddofs = [1:2*(nely+1):2*(nelx+1)*nely+2,2:2*(nely+1):2*(nelx+1)*nely+2];
end
U = zeros(2*(nely+1)*(nelx+1),1);
%% Fixeddofs rollers and pinned
alldofs = [1:2*(nely+1)*(nelx+1)];
freedofs = setdiff(alldofs,fixeddofs);
end