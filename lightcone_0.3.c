/*************************************************************
SimFast21
Calculates lightcone from standard Tb boxes.
Includes option to take into account evolution between redshift boxes.
Includes dz boxes consistent with RSD_LC
*************************************************************/
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#ifdef _OMPTHREAD_
#include <omp.h>
#endif
#include <complex.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <fftw3.h>     /* headers for FFTW library */

#include "Input_variables.h"
#include "auxiliary.h"

double drdz(double z);
double get_int_r(double z); 

double c_m = 2.9979e8; 
double pc = 3.08567758e16; //in m
double nu21 = 1420.0; //MHz
double k_b = 1.3806503e-23;
const double Tcmb0=2.725;       /* K */
const static int NR_END=1;

double *dvector(int, int);
void cubic_spline(double*, double *, int, int,
                  double , double, double *);
void setup_zD(double *DD, double *ZZ,double *DD2);
double cubic_splint(double *, double *, double *,
                    int , double );
void error(char error_text[]);
void free_dvector(double *, int , int );

int main(int argc, char * argv[]){
  get_Simfast21_params(argv[1]);
  char fname[256];
  float *temp;
  float fraction;
  double *t21,*t21dz;
  DIR* dir;
  double zmin,zmax,dz,zbox;
  double pi=3.14159;
  float* z_cell;
  float r_cell;
  //double nu1=050.0,nu2=160.0; //Frequencies of first and last observed maps
  double nu1=200.0,nu2=51.0;
  int n_maps; //Number of maps
  double del_r= global_dx_smooth;
  double del_nu_lc; // frequency resolution of light cone
  double d_box,r_i,z_i;
  int i,j,k;
  int kk;
  float i_r,j_r;
  double FoVdeg;
  double FoV;
  float* nu_p; // Frequency of photon as observed at telescope
  float* LC; // Intensity
  double *ZZ;
  double *DD,*DD2;
  int Dim; //Size of light cone
  long long cube_out_size;
  FILE *fid,*fid88;
  int i_sm,j_sm,k_sm; // indices in small box with boundary condt 
  double del_r_lc;
  int evo;
  
//*************************************************************************
  del_nu_lc = 0.1; // CHOOSE OUTPUT FREQUENCY SEPARATION (MHz)
  FoVdeg = 1.0; //in degrees   
  Dim = 400;
  evo = 0; // set to 1 if want to interpolate between two redshift boxes.
  //***************************************************************************
  FoV = (FoVdeg*pi)/180.0 ;// in radians
  
  /* opening directory for the output */
  sprintf(fname,"%s/Lightcone",argv[1]);
  if((dir=opendir(fname))==NULL) {
    printf("Creating lightcone directory\n");
    if(mkdir(fname,(S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH))!=0) {
      printf("Error creating directory!\n");
      exit(1);
    }
  }
  
sprintf(fname,"%s/Lightcone/frequency_event_lightcone_FoVdeg%4.2f_Dim%d.txt",argv[1],FoV,Dim);
  if((fid88 = fopen(fname,"w"))==NULL) {
    printf("Cannot open file:%s...\n",fname);
    exit(1);
  }
 
 
  ZZ=dvector(1,30000);
  DD=dvector(1,30000);DD2=dvector(1,30000);
  setup_zD(ZZ,DD,DD2);
  
  /* Check for correct number of parameters*/
  if(argc == 1 || argc > 5) {
    printf("Usage : intensity_code base_dir [zmin] [zmax] [dz]\n");
    exit(1);
  }
  
  /* getting the minimum and maximum z of the boxes */
  if(argc > 2) {
    zmin=atof(argv[2]);
    if (zmin < global_Zminsim) zmin=global_Zminsim;
    if(argc > 3) {
      zmax=atof(argv[3]);
      if(zmax>global_Zmaxsim) zmax=global_Zmaxsim;
      if(argc==5) dz=atof(argv[4]); else dz=global_Dzsim;
    }else {
      zmax=global_Zmaxsim;
      dz=global_Dzsim;
    }
    
    zmin=zmax-dz*ceil((zmax-zmin)/dz); /* make sure (zmax-zmin)/dz is an integer so that we get same redshifts starting from zmin or zmax...*/
  } else {
    zmin=global_Zminsim;
    zmax=global_Zmaxsim;
    dz=global_Dzsim;
  }
  
  d_box = get_int_r(global_Zmaxsim); // changed to global zmax 23/05
  zbox = zmax;
  //*********SET S,NU AND OMEGA RESOLUTIONS *********//  
  
  n_maps = floor((nu1 - nu2)/del_nu_lc)+1;
  printf("This code will produce %d maps spaced equally by %f MHz\n",n_maps,del_nu_lc);
  
  fflush(stdout); 
  if(!(temp=(float *) malloc(global_N3_smooth*sizeof(float)))) {
    printf("Problem1...\n");
    exit(1);
  }
  
  if(!(nu_p=(float *) malloc(n_maps*sizeof(float)))) {
    printf("Problem1...\n");
    exit(1);
  }

  if(!(z_cell=(float *) malloc(n_maps*sizeof(float)))) {
    printf("Problem1...\n");
    exit(1);
  }


  cube_out_size = n_maps*Dim*Dim;
  
  if(!(LC=(float *) malloc(cube_out_size*sizeof(float)))) {
    printf("Problem2...\n");
    exit(1);
  }
  
  if(!(t21=(double *) malloc(global_N3_smooth*sizeof(double)))) {
    printf("Problem...\n");
    exit(1);
  }
 if(!(t21dz=(double *) malloc(global_N3_smooth*sizeof(double)))) {
    printf("Problem...\n");
    exit(1);
  }
   
  r_i = d_box-del_r/2.0; // changed to minus 16/12
  
  z_i = cubic_splint(DD,ZZ,DD2,30000,r_i); // redshift of cell at k=0 
  
  for (int mm=0;mm<n_maps;mm++){
    
    nu_p[mm] = nu2 + (2.0*mm+1.0)*del_nu_lc/2.0; // middle of frequency bin being observed     
    z_cell[mm] = 1420.0/nu_p[mm]-1.0; // redshift of cell equivalent to observed frequency
  }
  
   /************ READ IN TB BOX  ***********/
  zbox=zmax-dz;
  for (kk=0;;kk++){
  if ((z_cell[0] < zbox) && (zbox-dz > zmin)){
    zbox = zbox-dz;}else break; } 
   printf("zbox:%f",zbox);
  sprintf(fname, "%s/deltaTb/deltaTb_z%.3f_N%ld_L%.0f.dat",argv[1],zbox,global_N_smooth,global_L); 
  fid=fopen(fname,"rb");
  if (fid==NULL) {
    printf("Error reading deltanl file... Check path or if the file exists..."); 
    exit (1);}
  fread(temp,sizeof(float),global_N3_smooth,fid);
  fclose(fid);

  for(i=0;i<global_N3_smooth;i++) t21[i] = temp[i]; 

  if (evo == 1) {
  sprintf(fname, "%s/deltaTb/deltaTb_z%.3f_N%ld_L%.0f.dat",argv[1],zbox+dz,global_N_smooth,global_L);
  fid=fopen(fname,"rb");
  if (fid==NULL) {
    printf("Error reading deltanl file... Check path or if the file exists..."); 
    exit (1);}
  fread(temp,sizeof(float),global_N3_smooth,fid);
  fclose(fid);

  for(i=0;i<global_N3_smooth;i++) t21dz[i] = temp[i];
}

/*****************************************************************************/


  for (int mm=0;mm<n_maps;mm++){
  fprintf(fid88,"mm: %d \t %fMHz\n",mm,nu_p[mm]);

  fflush(0);  
    r_cell = get_int_r(z_cell[mm]); // distance to cell
    k =  round((d_box - r_cell)/del_r); // pixel address
    k_sm = k;//pixel index equivalent in small box boundary conditions
    fraction = (z_cell[mm]-zbox)/dz;
    if (fraction  < 0.0) fraction=0.0;
    
    for (int jjj=0;;jjj++) if (k_sm<0)  k_sm+=global_N_smooth; else break;
    for (int jjj=0;;jjj++) if (k_sm>=global_N_smooth) k_sm-=global_N_smooth; else break;
 fprintf(fid88,"%f \t %f \t %f \t %f \t %f \t %d \t %d \n",r_cell,d_box,z_cell[mm],zbox,del_r,k_sm,k);
    fprintf(fid88,"k, k_sm, fraction, z_cell, z_box, z_box+dz: %d \t %d \t %f \t %f \t %f \t %f \n",k,k_sm,fraction,z_cell[mm],zbox,zbox+dz);
    fflush(0);
    if (evo == 1){
	if (fraction >= 0.5) fprintf(fid88,"main z_box:  %f \n",zbox+dz);
    fflush(0);
if (fraction < 0.5) fprintf(fid88,"main z_box: %f \n",zbox);
}


    for (int ii=0;ii<Dim;ii++){
      for (int jj=0;jj<Dim;jj++){
del_r_lc = r_cell * sin(FoV) / Dim;
              j_r = (jj - Dim/2) * del_r_lc ; // comoving distance in j direction
              j = floor(j_r/del_r)+global_N_smooth/2;

              j_sm = j;//pixel index equivalent in small box boundary conditions
              for (int jjj=0;;jjj++) if (j_sm<0)  j_sm+=global_N_smooth; else break;
              for (int jjj=0;;jjj++) if (j_sm>=global_N_smooth) j_sm-=global_N_smooth; else break;

              i_r = (ii - Dim/2) * del_r_lc ; // comoving distance in i direction
              i = floor(i_r/del_r)+global_N_smooth/2;
              i_sm = i;

              for (int jjj=0;;jjj++) if (i_sm<0)  i_sm+=global_N_smooth; else break;
              for (int jjj=0;;jjj++) if (i_sm>=global_N_smooth) i_sm-=global_N_smooth; else break;

if (evo == 0){
	LC[mm+n_maps*(jj+Dim*ii)] = t21[k_sm+global_N_smooth*(j_sm+global_N_smooth*i_sm)];
if (ii == 109 && jj==295) printf("here1: %d \t %d \t %f \t %f \t %d \t %e \n",ii,jj,zbox,z_cell[mm],k_sm,t21[k_sm+global_N_smooth*(j_sm+global_N_smooth*i_sm)]);	
} else {
LC[mm+n_maps*(jj+Dim*ii)] = t21[k_sm+global_N_smooth*(j_sm+global_N_smooth*i_sm)]*(1.-fraction) + t21dz[k_sm+global_N_smooth*(j_sm+global_N_smooth*i_sm)]*(fraction);
if (ii==109 && jj==295) printf("here: %e \t %e \t %f \t %f \t %d \t %e \n",t21[k_sm+global_N_smooth*(j_sm+global_N_smooth*i_sm)]*(1.-fraction),t21dz[k_sm+global_N_smooth*(j_sm+global_N_smooth*i_sm)]*(fraction),zbox,z_cell[mm],k_sm,t21[k_sm+global_N_smooth*(j_sm+global_N_smooth*i_sm)]);
}      
}
    }

  fflush(stdout);  
    if ((z_cell[mm+1] < zbox) && (zbox-dz > zmin)){
      zbox = zbox-dz;
      /************ READ IN TB BOX  ***********/
if (evo == 0 ){
      sprintf(fname, "%s/deltaTb/deltaTb_z%.3f_N%ld_L%.0f.dat",argv[1],zbox,global_N_smooth,global_L);

      fid=fopen(fname,"rb");
      if (fid==NULL) {
	printf("Error reading deltanl file... Check path or if the file exists...");
	exit (1);}
      fread(temp,sizeof(float),global_N3_smooth,fid);
      fclose(fid);
      
      for(i=0;i<global_N3_smooth;i++) t21[i] = temp[i];
} else {

for(i=0;i<global_N3_smooth;i++) t21dz[i] = t21[i];
sprintf(fname, "%s/deltaTb/deltaTb_z%.3f_N%ld_L%.0f.dat",argv[1],zbox,global_N_smooth,global_L);

      fid=fopen(fname,"rb");
      if (fid==NULL) {
        printf("Error reading deltanl file... Check path or if the file exists...");
        exit (1);}
      fread(temp,sizeof(float),global_N3_smooth,fid);
      fclose(fid);
 
for(i=0;i<global_N3_smooth;i++) t21[i] = temp[i];
}

							 /********************************************/	      

  fflush(stdout);					 
							 
     } 
    
  }// end of filling of this map
  
  
  
    sprintf(fname,"%s/Lightcone/deltaTb_LC_N%d_FOV%6.4f_dnu%4.2fMHz.dat",argv[1],Dim,FoVdeg,del_nu_lc);
 if (evo ==1 ) sprintf(fname,"%s/Lightcone/deltaTb_LC_N%d_FOV%6.4f_dnu%4.2fMHz_evo.dat",argv[1],Dim,FoVdeg,del_nu_lc);

    if((fid = fopen(fname,"wb"))==NULL) {
      printf("Cannot open file:%s...\n",fname);
      exit(1);
}    
    
    fwrite(LC,sizeof(float),cube_out_size,fid);
    fclose(fid);
    
    
  return 0;
}


/**************************************************************************************************************************************************************//* get r(z) in Mpc/h*/
double get_int_r(double z) {
    
    double dz=0.001,r;
    int n,i;

    n=(int)(z/dz)+1;
    dz=z/n;

    r=0.;
    for(i=0; i<n;i++) {
        r+=drdz(i*dz+dz/2.);
    }
 
    return r*dz;
}

/* dr/dz in comoving Mpc/h */

double drdz(double z) {

    return 2997.9/sqrt(global_omega_m*(1.+z)*(1.+z)*(1.+z)+global_lambda);  /* value in Mpc/h */


}

void free_dvector(double *v, int nl, int nh){
  free((char*) (v+nl-NR_END));
    //std::cout<<v<<"\n"<<std::flush;
    //  //delete[] v;
}

double *dvector(int nl, int nh){
  double *v;
    //static int i;
    //  //i++;
    //    //std::cout<<"nl is\t"<<nl<<"nh is\t"<<nh<<"\t"<<i<<"\t"<<v<<"\n"<<std::flush;
    //
      v=(double *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(double)));
    //        //v = new double [nh+1];
    //          //if (!v) tools::error((char *)"allocation failure in tools::dvector()");
             return v-nl+NR_END;
    //             
    //               //return v;
   }

void error(char error_text[]){

    fprintf(stderr,"Standard run-time error...\n");
      fprintf(stderr,"%s\n",error_text);
        fprintf(stderr,"...now exiting to system...\n");
	  //std::cin>>a; exit(1); //EOC DOES NOT WORK
	  }


void cubic_spline(double *x, double *y, int anf, int n,
		  double yp1, double ypn, double *y2){
	int i,k; double p,qn,sig,un,*u;
	u = dvector(1,n-1);
	if (yp1 > 0.99e30) {y2[anf]=u[anf]=0.0;}
		 else {
			y2[anf] = -0.5;
			u[anf]=(3.0/(x[anf+1]-x[anf]))*((y[anf+1]-y[anf])/(x[anf+1]-x[anf])-yp1);
		}
		for (i=anf+1;i<n;i++) {
		   sig=(x[i]-x[i-1])/(x[i+1]-x[i-1]);
		   p=sig*y2[i-1]+2.0;
		   y2[i]=(sig-1.0)/p;
		   u[i]=(y[i+1]-y[i])/(x[i+1]-x[i]) - (y[i]-y[i-1])/(x[i]-x[i-1]);
		   u[i]=(6.0*u[i]/(x[i+1]-x[i-1])-sig*u[i-1])/p;
		}
		if (ypn > 0.99e30) {qn=un=0.0;}
		else {
		qn=0.5;
		un=(3.0/(x[n]-x[n-1]))*(ypn-(y[n]-y[n-1])/(x[n]-x[n-1]));
		}
		y2[n]=(un-qn*u[n-1])/(qn*y2[n-1]+1.0);
		for (k=n-1;k>=anf;k--)
		y2[k]=y2[k]*y2[k+1]+u[k];
		free_dvector(u,1,n-1);
}

double cubic_splint(double *xa, double *ya, double *y2a,
		    int n, double x){
   int klo,khi,k;
   double h,b,a,y;
   klo=1;
   khi=n;
   while (khi-klo > 1){
       k=(khi+klo) >> 1;
       if (xa[k] > x) khi=k;
       else klo=k;
   }
   h=xa[khi]-xa[klo];
  if (h == 0.0) {
  //std::cerr<<"Value of x is:"<<x<<"\n"; //EOC DOES NOT WORK
  error((char *)"Bad xa input to routine spline::cubic_splint");
  }
  a=(xa[khi]-x)/h;
  b=(x-xa[klo])/h;
  y=a*ya[klo]+b*ya[khi]+((a*a*a-a)*y2a[klo]+(b*b*b-b)*y2a[khi])*(h*h)/6.0;
  return y;
}

void setup_zD(double *ZZ, double *DD,double *DD2){

int i=1;
for (double z=0;z<30;z+=0.001){

ZZ[i]=z;
DD[i]=get_int_r(z);
i++;
//printf("%e\t%e\t%e\t\n",z,ZZ[i],DD[i]);
}
cubic_spline(DD,ZZ,1,30000,1.0e30,1.0e30,DD2);

}
