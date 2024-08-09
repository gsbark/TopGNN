%%% Modified top110 code (E. Andreassen, A. Clausen, M. Schevenels,B. S. Lazarov and O. Sigmund )

function [num_iter,fconv,NN_c] = top110NN(nelx,nely,volfrac,penal,rmin,ft,holes,maxiter,eps,NN)
    %% MATERIAL PROPERTIES
    E0 = 1;
    Emin = 1e-5;
    nu = 0.3;
    %% PREPARE FINITE ELEMENT ANALYSIS
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
    % Load node 
    F = sparse(2*(nely+1)*(nelx+1),1,-1,2*(nely+1)*(nelx+1),1);
    % Fixeddofs all left
    fixeddofs = [1:2*nely+1];
    U = zeros(2*(nely+1)*(nelx+1),1);
    %% Fixeddofs rollers and pinned
    alldofs = [1:2*(nely+1)*(nelx+1)];
    freedofs = setdiff(alldofs,fixeddofs);
    el_dofs = 8;
    %% PREPARE FILTER
    iH = ones(nelx*nely*(2*(ceil(rmin)-1)+1)^2,1);
    jH = ones(size(iH));
    sH = zeros(size(iH));
    k = 0;
    for i1 = 1:nelx
      for j1 = 1:nely
        e1 = (i1-1)*nely+j1;
        for i2 = max(i1-(ceil(rmin)-1),1):min(i1+(ceil(rmin)-1),nelx)
          for j2 = max(j1-(ceil(rmin)-1),1):min(j1+(ceil(rmin)-1),nely)
            e2 = (i2-1)*nely+j2;
            k = k+1;
            iH(k) = e1;
            jH(k) = e2;
            sH(k) = max(0,rmin-sqrt((i1-i2)^2+(j1-j2)^2));
          end
        end
      end
    end
    H = sparse(iH,jH,sH);
    Hs = sum(H,2);
    %% INITIALIZE ITERATION
    x = repmat(volfrac,nely,nelx);
    %% Add Holes 
    passive = zeros(nely,nelx);
    if holes == 1
        for i = 1:nelx
            for j = 1:nely
                if j<nely/2 && i>nelx/2
                    passive(j,i) = 1;
                end
            end
        end
    end
    beta = 1;
    if ft == 1 || ft == 2
      xPhys = x;
    elseif ft == 3
      xTilde = x;
      xPhys = 1-exp(-beta*xTilde)+xTilde*exp(-beta);
    end
    loopbeta = 0;
    loop = 0;
    change = 1;
    %% START ITERATION
    run_NN =NN;
    cond_1 = 0;
    cond_2 = 0;
    cond_4 = 0;
    eps_start = eps;
    eps_conv = 2e-2;
    final_conv = 1e-5;
    GNN_cooldown = inf;
    %Create coords file
    history = zeros(nelx*nely,5);
    X = reshape(repmat(linspace(1, nelx, nelx), nely, 1),[],1);
    Y = reshape(repmat(flip(linspace(1, nely, nely)), nelx, 1)', [], 1);
    [res,modl] = pyrunfile("Initialize.py",["idx","model"],X =X,Y=Y);
    Obj = zeros(maxiter,1);
    Grays = ones(maxiter,1)*nelx*nely;
    stop=0;
    while change > 0.0005 && loop <maxiter
      tic
      loopbeta = loopbeta+1;
      loop = loop+1;
      GNN_cooldown = GNN_cooldown+1;
      G_curr = sum(xPhys(:) >= 0.05 & xPhys(:) <= 0.95);
      cond_3 = GNN_cooldown>=6;
      if loop >1 
          cond_4 = (Grays(loop-1)-G_curr)/Grays(loop-1) >eps_conv; 
      end 
      if cond_1 && cond_3 && cond_4  
          probs= double(pyrunfile("run_GNN.py","probs",model=modl,density=history,index=res));
          preds = probs;
          preds = reshape(preds,nely,nelx);
          x = preds;
          xPhys(:) = (H*x(:))./Hs;  
          GNN_cooldown=0;
      end
     
      if cond_2
          %Volume-preserving thresholding
          xPhys = Vol_thresholding(xPhys,volfrac,nelx,nely);
          stop=1;
          change = 0;
      end
    
      %% FE-ANALYSIS
      sK = reshape(KE(:)*(Emin+xPhys(:)'.^penal*(E0-Emin)),el_dofs^2*nelx*nely,1);
      K = sparse(iK,jK,sK);
      K = (K+K')/2;
      U(freedofs) = K(freedofs,freedofs)\F(freedofs);
      %% OBJECTIVE FUNCTION AND SENSITIVITY ANALYSIS
      ce = reshape(sum((U(edofMat)*KE(1:8,1:8)).*U(edofMat),2),nely,nelx);
      c = sum(sum((Emin+xPhys.^penal*(E0-Emin)).*ce));
      dc = -penal*(E0-Emin)*xPhys.^(penal-1).*ce;
      dv = ones(nely,nelx);
      Obj(loop)=c;
      Grays(loop)=sum(xPhys(:) >= 0.05 & xPhys(:) <= 0.95);

      cond_1 = Grays(loop)/(size(x,1)*size(x,2))<eps_start;
      if loop>1 
          cond_2 = abs((Obj(loop-1)-Obj(loop))/Obj(loop-1))< final_conv; 
      end
      %% FILTERING/MODIFICATION OF SENSITIVITIES
      if stop==0
          if ft == 1
            dc(:) = H*(x(:).*dc(:))./Hs./max(1e-3,x(:));
          elseif ft == 2
            dc(:) = H*(dc(:)./Hs);
            dv(:) = H*(dv(:)./Hs);
          elseif ft == 3
            dx = beta*exp(-beta*xTilde)+exp(-beta);
            dc(:) = H*(dc(:).*dx(:)./Hs);
            dv(:) = H*(dv(:).*dx(:)./Hs);
          end
          %% OPTIMALITY CRITERIA UPDATE OF DESIGN VARIABLES AND PHYSICAL DENSITIES
          l1 = 0; l2 = 1e9; move = 0.2;
          while (l2-l1)/(l1+l2) > 1e-3
            lmid = 0.5*(l2+l1);
            xnew = max(0,max(x-move,min(1,min(x+move,x.*sqrt(-dc./dv/lmid)))));
            if ft == 1
              xPhys = xnew;
            elseif ft == 2
              xPhys(:) = (H*xnew(:))./Hs;
            elseif ft == 3
              xTilde(:) = (H*xnew(:))./Hs;
              xPhys = 1-exp(-beta*xTilde)+xTilde*exp(-beta);
            end
            xPhys(passive==1) = 0;
            xPhys(passive==2) = 1;
            if sum(xPhys(:)) > volfrac*nelx*nely, l1 = lmid; else l2 = lmid; end
          end
          change = max(abs(xnew(:)-x(:)));
          x = xnew;
      end
      %% PRINT RESULTS
      fprintf(' It.:%5i Obj.:%11.4f Vol.:%7.3f ch.:%7.3f penal:%7.3f \n ' ,loop,c, ...
        mean(xPhys(:)),change,penal);
      %% PLOT DENSITIES
      colormap(gray); imagesc(1-xPhys); clim([0 1]); axis equal; axis off; drawnow;
      %% UPDATE HEAVISIDE REGULARIZATION PARAMETER
      if ft == 3 && beta < 64 && (loopbeta >= 50 || change <= 0.01)
        beta = 2*beta;
        loopbeta = 0;
        change = 1;
        fprintf('Parameter beta increased to %g.\n',beta);
      end
      for i=1:4
        history(:, i) = history(:,i + 1);
      end
        history(:,5) = xPhys(:);
    end
end

function [xPhys] = Vol_thresholding(xPhys,volfrac,nelx,nely)
    [~,I] = sort(xPhys(:),'descend');
    vt = floor(((volfrac-0.001)*nelx*nely)/(1-0.001));
    xPhys(I(1:vt))=1;
    xPhys(I(vt+1:end))=0.001;
    xPhys=reshape(xPhys,nely,nelx);
end
