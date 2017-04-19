
/*This code takes real space simulation boxes and produces a light cone, simultaneously taking into account the peculiar velocities of the hydrogen atoms. The code interpolates between two redshift boxes in order to account for the evolution between discretised redshift boxes.

 This code was written by Emma Chapman, Imperial College London. Please contact e.chapman@imperial.ac.uk with any comments or questions.
 Please cite the accompanying paper () when publishing work which uses this code.
 
 USAGE:
 ./RSD_LC_v1.1.c workdir nu1 nu2 del_nu_lc FoV_deg [LC_Dim] [z1] [z2]
 workdir: Working directory containing box folders. Lightcone will be output here in folder "Lightcone"
 nu1: Frequency of first map in light cone (MHz)
 nu2: Frequency of last map in light cone (MHz)
 del_nu_lc: Frequency separation between light cone slices
 FoV_deg: Field of view of light cone in degrees
 LC_Dim: Number of pixels along side of light cone map (if not specificied assumed same as input box dimension)
 z1: The lowest redshift box to include
 z2: The highest redshift box to include
 
 INPUT BOXES:
 The box formats are expected to be in the same format as output by SimFast21, i.e. binary files with c ordering.
 * = only if TS included in T21 calculation
 () = name of box as output by simfast21

 ionization (xHII)
 non-linear density field (deltanl)
 IGM temperature (TempX)*
 collisional coupling (xc)*
 Lyman-alpha coupling (xalpha)*
 velocity field (vel)
 
 OUTPUT:
 A 3D fits file of brightness temperature in units of K with constant field of view accross a chosen frequency range.
 

 Example:
 mpirun -np 4 ./RSD_LC /Users/echapman/Documents/Data_Cubes/Simfast_200Mpc_z27_z6/Original 142.1 142.0 0.1 1.0 400 8.0 10.0
 
 *************************************************************/
#include <mpi.h>
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

float cell_interp(float* box, float del_r, float fraction_i, float fraction_j, int i_sm, int j_sm, int k_sm);
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
    int numProcs, myid;
    MPI_Init (&argc,&argv);
    MPI_Comm_rank (MPI_COMM_WORLD, &myid);
    MPI_Comm_size (MPI_COMM_WORLD, &numProcs);
    get_Simfast21_params(argv[1]);
    //*******************************USER CHANGEABLE PARAMETERS*********************************
    double del_nu_lc,FoVdeg,nu1,nu2;
    int Dim;
    del_nu_lc = atof(argv[4]); // CHOOSE OUTPUT FREQUENCY SEPARATION (MHz)
    FoVdeg = atof(argv[5]); //in degrees
    nu1=atof(argv[2]);
    nu2=atof(argv[3]); //Frequencies of first and last observed maps
    if (argc > 6) {
        Dim=atoi(argv[6]);}
    else{
        Dim=global_N_smooth;
    }
    //******************************************************************************************
    //*******************************ADVANCED USER CHANGEABLE PARAMETERS************************
    //**********careful of overwriting: output filenames will not automatically change**********
    int pv,oneevent,dnu_off,evo;
    float dsdiv;
    pv=1; // SET TO ZERO IF WANT TO SET PV,DV=0
//    dsdiv = 10; // number to divide ds by to get smaller integration
    oneevent= 0; // set to one if want intital intensity of cell always to be CMB
    dnu_off = 0; // set to 1 if want to set d_nu = 0
    evo=1; //set to 1 to interpolate between z, z+dz boxes according to z_cell
    //******************************************************************************************
    
    char fname[256];
    FILE *fid;
    float *temp;
    double Icmb,Tcmb,xtot,aver,xtotdz;
    float *t21,*t21dz;
    float v_c_interp,v_cdz_interp,nHI_interp,nHIdz_interp,t21_interp,t21dz_interp,TS_interp,TSdz_interp;
    float dvds_H_interp, dvds_Hdz_interp;
    float fraction_j,fraction_i;
    float nHI_pix,t21_pix,TS_pix;
    float v_c_i;//initial peculiar velocity
    double del_r_lc;
    double zmin,zmax,dz,z;
    double pi=3.14159;
    double z_cell=0.0;
    int n_maps; //Number of maps
    double del_r= global_dx_smooth;
    double del_s,del_s_pix;
    double d_z; // comoving distance to redshift z
    int mm_min,mm_max;
    double del_omega;// angular separation of lightcone cells.
    double del_nu_21;
    double nu_temp;
    double nu_i; // Frequency of photon entering patch 
    double nu_f; // Frequency of photon leaving patch 
    double nu_min, nu_max; // Define intersection of frequency patch with line width
    double del_nu; // Change in frequency of photon traversing patch ds
    double d_box,del_I,r,r_i,z_i;
    double i_r,j_r,k_r;
    int i,j,k,ij;
    int i_sm,j_sm,k_sm; // indices in small box with boundary condt
    double FoV;
    double nu_p; // Frequency of photon as observed at telescope
    float* v_c; // Peculiar velocity cube divided by c
    float* v_cdz; // Peculiar velocity cube divided by c
    float* dvds_H; // dv/ds divided by H
    float* dvds_Hdz; // dv/ds divided by H
    float *nHI,*nHIdz,*TS,*TSdz;
    float v_c_pix, dvds_H_pix; // pixel values of v_c, dvds_H
    float* I21; // Intensity
    float* all_I21; // Intensity merged from MPI multiple processors
    double c_Mpc_h; // speed of light in Mpc/h/s
    double *ZZ,*DD,*DD2;
    float *nu_last, *all_nu_last;
    float* nu_first;
    long long LC_size;
    double fraction;
    double Tcmbdz;
    long *num,*all_num;
    float del_s_nu21;
    float ave=0.0;
    long long aven=0;
    float zevent=0.0;    
    FoV = (FoVdeg*pi)/180.0 ;// in radians
    
    
    /* Check for correct number of parameters*/
    if(argc == 1 || argc > 10) {
        printf("Usage :  RSD_LC_v1.0.c workdir nu1 nu2 del_nu_lc FoV_deg [LC_Dim] [zmin] [zmax] [dz]\n");
        exit(1);
    }
    del_nu_21 = nu21/c_m * pow((2.0 * 10000.0 * 1.3806503e-23 / mbar) ,0.5);
    // del_nu_21 = del_nu_21 * 10.0; // TESTING ONLY
    c_Mpc_h = c_m/(pc*1.0e6)*global_hubble;
    
    /* getting the minimum and maximum z of the boxes */
    if(global_use_Lya_xrays==0) printf("Lya and xray use set to false - assuming TS>>TCMB in t21 calculation.\n");
    if(argc > 7) {
        zmin=atof(argv[7]);
        if (zmin < global_Zminsim) zmin=global_Zminsim;
        if(argc > 8) {
            zmax=atof(argv[8]);
            if(zmax>global_Zmaxsim) zmax=global_Zmaxsim;
            if(argc==10) dz=atof(argv[9]); else dz=global_Dzsim;
        }else {
            zmax=global_Zmaxsim;
            dz=global_Dzsim;
        }
        zmin=zmax-dz*ceil((zmax-zmin)/dz); /* make sure (zmax-zmin)/dz is an integer so that we get same redshifts starting from zmin or zmax...*/
    } else {
      zmin = floor(nu21/(nu1)-1.)-dz;
      zmax = ceil(nu21/(nu2)-1.)+dz;
        dz=global_Dzsim;
    }
    
    d_box = get_int_r(zmax);
    r_i = d_box-del_r/2.0;
    // Set up arrays for z to r interpolation
    ZZ=dvector(1,30000);
    DD=dvector(1,30000);DD2=dvector(1,30000);
    setup_zD(ZZ,DD,DD2);
    z_i = cubic_splint(DD,ZZ,DD2,30000,r_i); // redshift of cell at first point in box along line of sight
    if (myid==0) printf("Here 6.11 r_i: %f z_i: %f \t d_box: %f \n",r_i,z_i,d_box);
    //*********SET NU AND OMEGA RESOLUTIONS *********//
    n_maps = floor((nu1 - nu2)/del_nu_lc)+1;
    if (myid==0) printf("This code will produce %d maps spaced equally by %f MHz\n",n_maps,del_nu_lc);
    del_omega = FoV/(Dim);
    LC_size = n_maps*Dim*Dim;
    
    if(!(temp=(float *) malloc(global_N3_smooth*sizeof(float)))) {
        printf("Problem allocating memory to arrays...\n");
        exit(1);
    }
    if(!(nHI=(float *) malloc(global_N3_smooth*sizeof(float)))) {
        printf("Problem allocating memory to arrays...\n");
        exit(1);
    }
    if(!(nHIdz=(float *) malloc(global_N3_smooth*sizeof(float)))) {
        printf("Problem allocating memory to arrays...\n");
        exit(1);
    }
    if(!(v_c=(float *) malloc(global_N3_smooth*sizeof(float)))) {
        printf("Problem allocating memory to arrays...\n");
        exit(1);
    }
    if(!(v_cdz=(float *) malloc(global_N3_smooth*sizeof(float)))) {
        printf("Problem allocating memory to arrays...\n");
        exit(1);
    }
    if(!(dvds_H=(float *) malloc(global_N3_smooth*sizeof(float)))) {
        printf("Problem allocating memory to arrays...\n");
        exit(1);
    }
    if(!(dvds_Hdz=(float *) malloc(global_N3_smooth*sizeof(float)))) {
        printf("Problem allocating memory to arrays...\n");
        exit(1);
    }
    if(!(I21=(float *) malloc(LC_size*sizeof(float)))) {
        printf("Problem allocating memory to arrays...\n");
        exit(1);
    }
    if(!(nu_last=(float *) malloc(LC_size*sizeof(float)))) {
        printf("Problem allocating memory to arrays...\n");
        exit(1);
    }
    if(!(all_nu_last=(float *) malloc(LC_size*sizeof(float)))) {
        printf("Problem allocating memory to arrays...\n");
        exit(1);
    }
    
    if(!(nu_first=(float *) malloc(LC_size*sizeof(float)))) {
        printf("Problem allocating memory to arrays...\n");
        exit(1);
    }
    if(!(t21=(float *) malloc(global_N3_smooth*sizeof(float)))) {
        printf("Problem allocating memory to arrays...\n");
        exit(1);
    }
    if(!(t21dz=(float *) malloc(global_N3_smooth*sizeof(float)))) {
        printf("Problem allocating memory to arrays...\n");
        exit(1);
    }

    if(!(TS=(float *) malloc(global_N3_smooth*sizeof(float)))) {
        printf("Problem allocating memory to arrays...\n");
        exit(1);
    }
    if(!(TSdz=(float *) malloc(global_N3_smooth*sizeof(float)))) {
        printf("Problem allocating memory to arrays...\n");
        exit(1);
    }
    
    for(int ind=0;ind<n_maps*Dim*Dim;ind++) I21[ind]=0.0;//SO ALL ZERO FOR MERGING OF MPI ARRAYS

    if(!(all_I21=(float *) malloc(LC_size*sizeof(float)))) {
        printf("Problem allocating memory to arrays...\n");
        exit(1);
    }
    
    // TEST ARRAYS - TO BE DELETED IN FINAL VERSION
    if(!(num=(long *) malloc(LC_size*sizeof(long)))) {
        printf("Problem allocating memory to arrays...\n");
        exit(1);
    }
    
    for(int ind=0;ind<n_maps*Dim*Dim;ind++) num[ind]=0.0;
    
    if(!(all_num=(long *) malloc(LC_size*sizeof(long)))) {
        printf("Problem allocating memory to arrays...\n");
        exit(1);
    }
    
    if (myid==0) printf("Here3 %f \t %f \t %f \n",zmax,zmin,dz);
    del_s = (c_Mpc_h/Hz(z))*(dz/(1.0+zmax));
        del_s_pix = del_r/(1.0+zmax);
        del_s_nu21 = (get_int_r(nu21/(nu21-del_nu_21)-1.0) - get_int_r(nu21/(nu21+del_nu_21)-1.0))/(1.0+zmax); // for testing 4/2/16
        if (del_s < del_s_nu21) del_s = del_s_nu21;
        if (del_s_pix < del_s) del_s = del_s_pix; // so always one integration point per pixel

	 if (myid==0) printf("This code will step with ds %f \n",del_s);
    /**************************************************/
    /************** redshift cycle ********************/
    /**************************************************/
    for(z=zmax-dz;z>(zmin-dz/10);z-=dz){
        // for(z=9.0;z>(8.6);z-=dz){
	zevent=0.0;
      
        if (myid==0) printf("Filling frequency maps from box z = %f\n",z);fflush(0);
       
        /************ READ IN DENSITY AND IONIZATION FIELDS TO CALC nHI  ***********/
        sprintf(fname, "%s/delta/deltanl_z%.3f_N%ld_L%.0f.dat",argv[1],z,global_N_smooth,global_L);
        fid=fopen(fname,"rb");
        if (fid==NULL) {
            printf("Error opening file:%s\n",fname);
            exit (1);}
        fread(temp,sizeof(float),global_N3_smooth,fid);
        fclose(fid);
        for(i=0;i<global_N3_smooth;i++) nHI[i] = (1.0+temp[i]);
        
        sprintf(fname, "%s/delta/deltanl_z%.3f_N%ld_L%.0f.dat",argv[1],z+dz,global_N_smooth,global_L);
        fid=fopen(fname,"rb");
        if (fid==NULL) {
            printf("Error opening file:%s\n",fname);
            exit (1);}
        fread(temp,sizeof(float),global_N3_smooth,fid);
        fclose(fid);
        
        for(i=0;i<global_N3_smooth;i++) nHIdz[i] = (1.0+temp[i]);
        
        sprintf(fname,"%s/Ionization/xHII_z%.3f_eff%.2lf_N%ld_L%.0f.dat",argv[1],z,global_eff,global_N_smooth,global_L);
        if((fid = fopen(fname,"rb"))==NULL) {
            printf("Error opening file:%s\n",fname);
            exit(1);
        }
        fread(temp,sizeof(float),global_N3_smooth,fid);
        fclose(fid);
        for(i=0;i<global_N3_smooth;i++) nHI[i] =  nHI[i] * (1. - temp[i]) * global_omega_b * 3.0 * pow(H0 * global_hubble,2.0) / (8.0 * PI * G) * pow(1.0+z,3.0) * 0.76 / mbar;
        
        sprintf(fname,"%s/Ionization/xHII_z%.3f_eff%.2lf_N%ld_L%.0f.dat",argv[1],z+dz,global_eff,global_N_smooth,global_L);
        if((fid = fopen(fname,"rb"))==NULL) {
            printf("Error opening file:%s\n",fname);
            exit(1);
        }
        fread(temp,sizeof(float),global_N3_smooth,fid);
        fclose(fid);
        for(i=0;i<global_N3_smooth;i++) nHIdz[i] =  nHIdz[i] * (1. - temp[i]) * global_omega_b * 3.0 * pow(H0 * global_hubble,2.0) / (8.0 * PI * G) * pow(1.0+z+dz,3.0) * 0.76 / mbar;
        /************** READ IN XRAY FILES IF USING TS ****************************/
        if(z>(global_Zminsfr-global_Dzsim/10) && global_use_Lya_xrays==1) {
	  Tcmb=Tcmb0*(1.+z);
	  Tcmbdz=Tcmb0*(1.+z+dz);
	  sprintf(fname,"%s/x_c/xc_z%.3lf_N%ld_L%.0f.dat",argv[1],z,global_N_smooth,global_L);
	  if((fid = fopen(fname,"rb"))==NULL) {
	    printf("Error opening file:%s\n",fname);
	    exit(1);
	  }
	  fread(temp,sizeof(float),global_N3_smooth,fid);
	  fclose(fid);
	  for(i=0;i<global_N3_smooth;i++) t21[i]=(double)temp[i];
          
	  sprintf(fname,"%s/x_c/xc_z%.3lf_N%ld_L%.0f.dat",argv[1],z+dz,global_N_smooth,global_L);
	  if((fid = fopen(fname,"rb"))==NULL) {
	    printf("Error opening file:%s\n",fname);
	    exit(1);
	  }
	  fread(temp,sizeof(float),global_N3_smooth,fid);
	  fclose(fid);
	  for(i=0;i<global_N3_smooth;i++) t21dz[i]=(double)temp[i];
          
	  sprintf(fname,"%s/Lya/xalpha_z%.3lf_N%ld_L%.0f.dat",argv[1],z,global_N_smooth,global_L);
	  if((fid = fopen(fname,"rb"))==NULL) {
	    printf("Error opening file:%s\n",fname);
	    exit(1);
	  }
	  fread(temp,sizeof(float),global_N3_smooth,fid);
	  fclose(fid);
	  aver=0.;
	  for(i=0;i<global_N3_smooth;i++) {
	    xtot=(double)temp[i]+t21[i];
	    t21[i]=xtot/(1.+xtot);
	  }
	  sprintf(fname,"%s/Lya/xalpha_z%.3lf_N%ld_L%.0f.dat",argv[1],z+dz,global_N_smooth,global_L);
	  if((fid = fopen(fname,"rb"))==NULL) {
	    printf("Error opening file:%s\n",fname);
	    exit(1);
	  }
	  fread(temp,sizeof(float),global_N3_smooth,fid);
	  fclose(fid);
          
	  aver=0.;
	  for(i=0;i<global_N3_smooth;i++) {
	    xtotdz=(double)temp[i]+t21dz[i];
	    t21dz[i]=xtotdz/(1.+xtotdz);
	  }
          
	  sprintf(fname,"%s/xrays/TempX_z%.3lf_N%ld_L%.0f.dat",argv[1],z,global_N_smooth,global_L);
	  if((fid = fopen(fname,"rb"))==NULL) {
	    printf("Error opening file:%s\n",fname);
	    exit(1);
	  }
	  fread(temp,sizeof(float),global_N3_smooth,fid);
	  fclose(fid);
	  
	  for(i=0;i<global_N3_smooth;i++) {
	    t21[i]=t21[i]*(1.-Tcmb/(double)temp[i]); // Temperature correction for high redshifts
	    TS[i]=Tcmb/(1.-t21[i]);
	  }
            
            sprintf(fname,"%s/xrays/TempX_z%.3lf_N%ld_L%.0f.dat",argv[1],z+dz,global_N_smooth,global_L);
            if((fid = fopen(fname,"rb"))==NULL) {
                printf("Error opening file:%s\n",fname);
                exit(1);
            }
            fread(temp,sizeof(float),global_N3_smooth,fid);
            fclose(fid);

            for(i=0;i<global_N3_smooth;i++) {
                t21dz[i]=t21dz[i]*(1.-Tcmbdz/(double)temp[i]); // Temperature correction for high redshifts
                TSdz[i]=Tcmbdz/(1.-t21dz[i]);
            }
        }else {
            for(i=0;i<global_N3_smooth;i++) t21[i]=1.0;
            for(i=0;i<global_N3_smooth;i++) t21dz[i]=1.0;
        }
        
        /************** READ IN VELOCITY AND VELOCITY GRADIENT BOXES ***************/
        sprintf(fname, "%s/Velocity/vel_z%.3f_N%ld_L%.0f_3.dat",argv[1],z,global_N_smooth,global_L);
        fid=fopen(fname,"rb");
        if (fid==NULL) {printf("Error reading vel file... Check path or if the file exists...");  printf("Path: %s",fname);exit (1);}
        fread(temp,sizeof(float),global_N3_smooth,fid);
        fclose(fid);
        for(i=0;i<global_N3_smooth;i++) v_c[i] = temp[i]; // velocity in proper units divided by c
        sprintf(fname, "%s/Velocity/dvdr_z%.3f_N%ld_L%.0f_3.dat",argv[1],z,global_N_smooth,global_L);
        fid=fopen(fname,"rb");
        if (fid==NULL) {printf("Error reading dvdr file... Check path or if the file exists...");printf("Path: %s \n",fname); exit (1);}
        fread(temp,sizeof(float),global_N3_smooth,fid);
        fclose(fid);
        for(i=0;i<global_N3_smooth;i++) dvds_H[i] = temp[i]; // dvds divided by H
        
        sprintf(fname, "%s/Velocity/vel_z%.3f_N%ld_L%.0f_3.dat",argv[1],z+dz,global_N_smooth,global_L);
        fid=fopen(fname,"rb");
        if (fid==NULL) {printf("Error reading vel file... Check path or if the file exists..."); exit (1);}
        fread(temp,sizeof(float),global_N3_smooth,fid);
        fclose(fid);
        for(i=0;i<global_N3_smooth;i++) v_cdz[i] = temp[i]; // velocity in proper units divided by c
        sprintf(fname, "%s/Velocity/dvdr_z%.3f_N%ld_L%.0f_3.dat",argv[1],z+dz,global_N_smooth,global_L);
        fid=fopen(fname,"rb");
        if (fid==NULL) {printf("Error reading dvdr file... Check path or if the file exists..."); exit (1);}
        fread(temp,sizeof(float),global_N3_smooth,fid);
        fclose(fid);
        for(i=0;i<global_N3_smooth;i++) dvds_Hdz[i] = temp[i]; // dvds divided by H
        
        fflush(stdout);
        /************START LOOPING THROUGH angles and s****************************/
        for (int ii=0;ii<Dim;ii++){
//            if (myid==0) printf("ii: %d \n ",ii); fflush(0);
            for (int jj=0;jj<Dim;jj++){
                ij = ii+jj*Dim;
                if (ij % numProcs == myid){
                    mm_min = 0;
                    mm_max = n_maps-1;
                    for (int mm=0;mm<n_maps;mm++){
                        nu_p = nu2 + (2.0*mm+1.0)*del_nu_lc/2.0;
                        if (nu_p < nu21/(z+1.)-5.) {mm_min=mm;} // Only searches redshift boxes withing 5MHz range of direct frequency equivalent. Makes code much faster.
                        if (nu_p < nu21/(z+1.)+5.) {mm_max=mm;}
                    }
                    for (int mm=mm_min;mm<mm_max+1;mm++){
                        z_cell = z_i; // redshift of cell at k=0. Reinitialized for every map being filled.
                        r=r_i; //comoving distance of cell at k=0.
                        
                        nu_p = nu2 + (2.0*mm+1.0)*del_nu_lc/2.0; // middle of frequency bin being observed

                        for (int kk=0; ; kk++){
//			    if (kk>=303 && kk <= 455 && myid==0 && ii==396 && jj==283) printf("Here 1: %f  \t %f \t %f \n",nu_i,nu_f,del_nu);
                            /********************** what cell are we looking at? **********************/
                            if (z_cell < z ) {break;}
                            fraction=(z_cell-z)/(dz);
			    if (evo==0) fraction=0.0; // testing only
                            if (fraction  < 0.0) fraction=0.0;
                            k_r =  d_box - r; // comoving distance in k direction
                            k = round(k_r/del_r);
                            k_sm = k;// pixel index equivalent in small box boundary conditions
                            for (int jjj=0;;jjj++) if (k_sm<0)  k_sm+=global_N_smooth; else break;
                            for (int jjj=0;;jjj++) if (k_sm>=global_N_smooth) k_sm-=global_N_smooth; else break;
			    
                            del_r_lc = r * sin(FoV) / Dim;
                            
                            j_r = (jj - Dim/2) * del_r_lc ; // comoving distance in j direction
                            j = round(j_r/del_r)+global_N_smooth/2;
                            j_sm = j;// pixel index equivalent in small box boundary conditions
                            for (int jjj=0;;jjj++) if (j_sm<0)  j_sm+=global_N_smooth; else break;
                            for (int jjj=0;;jjj++) if (j_sm>=global_N_smooth) j_sm-=global_N_smooth; else break;
                            fraction_j = (round(j_r/del_r)-j_r/del_r);
                            
                            i_r = (ii - Dim/2) * del_r_lc ; // comoving distance in i direction
                            i = round(i_r/del_r)+global_N_smooth/2;
                            i_sm = i;
                            for (int jjj=0;;jjj++) if (i_sm<0)  i_sm+=global_N_smooth; else break;
                            for (int jjj=0;;jjj++) if (i_sm>=global_N_smooth) i_sm-=global_N_smooth; else break;
                            fraction_i = (round(i_r/del_r)-i_r/del_r);
			    
			
                            /******************* CALCULATE |V|, |DVDS|, nHI IN CELL ***************/
                            
                            v_c_interp = cell_interp(v_c,del_r,fraction_i,fraction_j,i_sm,j_sm,k_sm);
                            v_cdz_interp = cell_interp(v_cdz,del_r,fraction_i,fraction_j,i_sm,j_sm,k_sm);
                            v_c_pix =
                            (v_c_interp)*(1-fraction) +
                            (v_cdz_interp)*(fraction);
                            
                            dvds_H_interp = cell_interp(dvds_H,del_r,fraction_i,fraction_j,i_sm,j_sm,k_sm);
                            dvds_Hdz_interp = cell_interp(dvds_Hdz,del_r,fraction_i,fraction_j,i_sm,j_sm,k_sm);
                            dvds_H_pix =
                            (dvds_H_interp)*(1-fraction) +
                            (dvds_Hdz_interp)*(fraction);
                            
                            nHI_interp = cell_interp(nHI,del_r,fraction_i,fraction_j,i_sm,j_sm,k_sm);
                            nHIdz_interp = cell_interp(nHIdz,del_r,fraction_i,fraction_j,i_sm,j_sm,k_sm);
                            nHI_pix =
                            (nHI_interp)*(1-fraction) +
                            (nHIdz_interp)*(fraction);
                            
                            t21_interp = cell_interp(t21,del_r,fraction_i,fraction_j,i_sm,j_sm,k_sm);
                            t21dz_interp = cell_interp(t21dz,del_r,fraction_i,fraction_j,i_sm,j_sm,k_sm);
                            t21_pix =
                            (t21_interp)*(1-fraction) +
                            (t21dz_interp)*(fraction);

			    TS_interp = cell_interp(TS,del_r,fraction_i,fraction_j,i_sm,j_sm,k_sm);
                            TSdz_interp = cell_interp(TSdz,del_r,fraction_i,fraction_j,i_sm,j_sm,k_sm);
                            TS_pix =
			      (TS_interp)*(1-fraction) +
                            (TSdz_interp)*(fraction);
                            
                           // if (myid==0) printf("CiC: %e \t %f \t %f \t %f \t %e \n",t21[k_sm+global_N_smooth*(j_sm+global_N_smooth*i_sm)],del_r,fraction_i,fraction_j,t21_interp);

                            
                            if (pv==0){
                                v_c_pix=0;
                                dvds_H_pix=0;
                            }
                            /***************************************************************************/
			    if (kk==0) {
			      v_c_i = cell_interp(v_c,del_r,fraction_i,fraction_j,i_sm,j_sm,0);
			      nu_i = nu_p * (1.0 + z_i) * (1.0 + v_c_i);
			      Tcmb=Tcmb0*(1+z_i);
			      Icmb = Tcmb * 2.0*k_b*pow(nu_i,2.0)/pow(c_m,2.0)*1.0e12 * (1.0+3.0*v_c_i);
			    }
                            del_nu = nu_i * del_s/c_Mpc_h * Hz(z_cell) * (1.0 + dvds_H_interp);
                            nu_f = nu_i - del_nu;
                            if (dnu_off==1) del_nu = 0.0;
         
                            if (z_cell <= z+dz) // As otherwise no 21-cm event will be observed as photon arrives before/after our time.
                            {
                                /************Do we have a 21-cm event?**************/
	
			      if (!(((nu_i<(nu21-del_nu_21/2.0)) && (nu_f<(nu21-del_nu_21/2.0))) ||   ((nu_i>(nu21+del_nu_21/2.0)) && (nu_f>(nu21+del_nu_21/2.0))))) // added 10/2 to charcterise in terms of non-events
                                {
				zevent=1.0;
				  if (del_nu < 0){ // changed to _rest 10/2/16
				    nu_max = nu_f;
				    nu_temp = nu21+del_nu_21/2.0;
				    if (nu_max >= nu_temp) {nu_max = nu_temp;}
                                    
				    nu_min = nu_i;
				    nu_temp = nu21-del_nu_21/2.0;
				    if (nu_min <= nu_temp) {nu_min = nu_temp;}
				  }
				  else
                                    {
				      nu_max = nu_i;
				      nu_temp = nu21+del_nu_21/2.0;
				      if (nu_max >= nu_temp) {nu_max = nu_temp;}
                                      
				      nu_min = nu_f;
				      nu_temp = nu21-del_nu_21/2.0;
				      if (nu_min <= nu_temp) {nu_min = nu_temp;}
                                    }

				  if (nu_max == nu_min) del_I = 1.60137e-40 * nHI_pix / del_nu_21;
				  else if (nu_max < nu_min) del_I = 0.0;
				  else if (nu_max > nu_min) del_I = 1.60137e-40/(Hz(z_cell)*(1.0+dvds_H_pix)) *
							      nHI_pix* c_m * (nu_max - nu_min) / ((nu21*1.0e6) * del_nu_21);
//				  if (myid==0 && ii==396 && jj==283) printf("Here 2: %f \t %f \t %f  \t %f \t %f \n",v_c_i,dvds_H_interp,nu_i,nu_f,del_nu); 
				  if (myid==0 && ii==396 && jj==283) printf("%d  \t %e \t %e \t %e \t %e \t %e \t %e \t %e \t %e \n",kk ,nu_max -nu_min, nu_i,nu_f,nu21+del_nu_21/2.0,nu21-del_nu_21/2.0,del_I,del_s,del_nu_21); // for line_follower.m
				  // ADJUST FOR HIGH REDSHIFT
				  if(z>(global_Zminsfr-global_Dzsim/10) && global_use_Lya_xrays==1) {
				    if (oneevent==1) {
				      if (myid==0 && ii==396 && jj==283) printf(" %e \n",del_I);
				      del_I =  del_I * (1 - (Icmb / ((2*k_b/pow((c_m/1420.0e6),2.0)) * TS_pix)));
				  
				    }				    
				    else{
				      //	      if (myid==0 && ii==396 && jj==283) printf("HERE22 %e \t %e\n",del_I * (1 - (Icmb / ((2*k_b/pow((c_m/1420.0e6),2.0)) * TS_pix))),del_I * (1 - ((Icmb+I21[mm+n_maps*(jj+Dim*ii)]) / ((2*k_b/pow((c_m/1420.0e6),2.0)) * TS_pix))));				    
				      del_I =  del_I * (1 - ((Icmb+I21[mm+n_maps*(jj+Dim*ii)]) / ((2*k_b/pow((c_m/1420.0e6),2.0)) * TS_pix)));				      
				    }
				    
				  }
				  
				  //  del_I = del_I * pow((1.0-v_c_pix),3.0); wrong now?
				  
				    
				  // Add intensity to appropriate map
				  
				  I21[mm+n_maps*(jj+Dim*ii)] = I21[mm+n_maps*(jj+Dim*ii)]*(1.0 - 3.0 * del_s/c_Mpc_h * Hz(z_cell) * (1.0 - dvds_H_interp)) + del_I ;
				  nu_last[mm+n_maps*(jj+Dim*ii)] = nu_f; // records last frequency of event for final adjustment later.
				  if (num[mm+n_maps*(jj+Dim*ii)] == 0) nu_first[mm+n_maps*(jj+Dim*ii)] = nu_i;
				  
				  num[mm+n_maps*(jj+Dim*ii)] = num[mm+n_maps*(jj+Dim*ii)] + 1.0 ;
                                }
                            }
                            r = r - del_s * (1.0 + z_cell);
                            z_cell = z_cell - del_s *(1+z_cell)*Hz(z_cell)/c_Mpc_h; // z of next pixel
                            nu_i = nu_f;
//			    if (kk>=303 && kk <= 455 && myid==0 && ii==396 && jj==283) printf("Here 3: %f  \t %f \t %f \n",nu_i,nu_f,del_nu);
                        } // end of kk loop i.e. line of sight ended
                    }// end of filling of this map
		    //   if (ii==396 && jj==283) printf("here3: %ld \n",num[0+n_maps*(jj+Dim*ii)]);
                } // MPI loop
            } // end of jj loop
        } // end of ii loop
	//if (zevent==1.0) printf("Events at this redshift!"); 
    } // end of redshift cycle
    // CONVERT TO Tb
    
    // OUTPUT MAPS - ONLY FOR ONE PROCESSOR.
    MPI_Barrier(MPI_COMM_WORLD); // WAITS FOR ALL PROCESSORS TO FINISH
    MPI_Allreduce(I21,all_I21,LC_size,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD); // MERGES ALL I21 VALUES FROM EACH PROCESSOR
    MPI_Allreduce(num,all_num,LC_size,MPI_LONG,MPI_SUM,MPI_COMM_WORLD);
    MPI_Allreduce(nu_last,all_nu_last,LC_size,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD); // WAITS FOR ALL PROCESSORS TO FINISH MERGING
    
    if (myid==0){
      for (int mm=0;mm<n_maps;mm++)
	{
	  nu_p = nu2 + (2.0*mm+1.0)*del_nu_lc/2.0; // middle of frequency bin being observed
	  d_z = get_int_r(1420.0/nu_p - 1.0); //Distance to that redshift
	  
	  aven=0;
	  ave=0.;

	  for (int ii=0;ii<Dim;ii++)
	    {
	      for (int jj=0;jj<Dim;jj++)
		{
		  
		  //ADJUST TO OBSERVATION FREQUENCY
		  
		  if (all_num[mm+n_maps*(jj+Dim*ii)] != 0){
		    all_I21[mm+n_maps*(jj+Dim*ii)] = all_I21[mm+n_maps*(jj+Dim*ii)] * pow((nu_p/all_nu_last[mm+n_maps*(jj+Dim*ii)]),3.0)*c_m*c_m/(2.0*k_b*nu_p*nu_p*1.0e12);
		    
		  }
		  ave = ave+all_I21[mm+n_maps*(jj+Dim*ii)]; aven = aven+1;		  
		}
            }
        }
      
    }
   
    ave = ave/aven * 1000.;
    if (myid==0) printf("average pixel value (mK): %e \n",ave);
    
    // OUTPUT ALL MAPS WITH DEL_NU SEPARATION
    
    if (myid==0){
      sprintf(fname,"%s/deltaTb_RSD/LightconeRSD_N%d_FOV%06.4f_dnu%04.2fMHz_%06.2fMHz_%06.2fMHz_ds%08.6f_div%05.2f_pv%d_oneevent%d.dat",argv[1],Dim,FoVdeg,del_nu_lc,nu1,nu2,del_s,dsdiv,pv,oneevent);
        if((fid = fopen(fname,"wb"))==NULL) {
            printf("Cannot open file:%s...\n",fname);
            exit(1);
        }
        fwrite(all_I21,sizeof(float),LC_size,fid);
        fclose(fid);
    }
    
    if (myid==0){
      sprintf(fname,"%s/deltaTb_RSD/NUM_RSD_N%d_FOV%06.4f_dnu%04.2fMHz_%06.2fMHz_%06.2fMHz_ds%08.6f_div%05.2f_pv%d_oneevent%d.dat",argv[1],Dim,FoVdeg,del_nu_lc,nu1,nu2,del_s,dsdiv,pv,oneevent);
        if((fid = fopen(fname,"wb"))==NULL) {
            printf("Cannot open file:%s...\n",fname);
            exit(1);
        }
        fwrite(all_num,sizeof(long),LC_size,fid);
        fclose(fid);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);  
    MPI_Finalize ();
    
    return 0;
}


/**************************************************************************************************************************************************************/
// Cloud in Cells 
// BUGGY POSSIBLY. OVERWRITTEN TO JUST USE ORIGINAL CELLVALUE AT BOTTOM OF FUNCTION UNTIL HAVE TIME TO CHECK
float cell_interp(float* box, float del_r, float fraction_i ,float fraction_j, int i_sm, int j_sm, int k_sm){
    
    float val;
    float cell_x,cell_y,cell_xy,cell_c;
    int i2=10000,j2=10000;
    float fraction_jj,fraction_ii;
    
    fraction_ii = fabs(fraction_i)*del_r;
    fraction_jj = fabs(fraction_j)*del_r;
    
    cell_c = (del_r - fraction_ii)*(del_r - fraction_jj)/pow(del_r,2.0);
    cell_x = (fraction_ii)*(del_r - fraction_jj)/pow(del_r,2.0);
    cell_y = (del_r - fraction_ii)*(fraction_jj)/pow(del_r,2.0);
    cell_xy = (fraction_ii)*(fraction_jj)/pow(del_r,2.0);
    
    
   /*orig if (fraction_j >= 0) {
        if (j_sm==0)  j2+=global_N_smooth-1; 
        else j2 = j_sm-1;
    }else
    {
        if (j_sm==global_N_smooth-1)  j2-=global_N_smooth-1;
        else j2 = j_sm+1;
    }
    
    if (fraction_i >= 0) {
        if (i_sm==0)  i2+=global_N_smooth-1; 
        else i2 = i_sm-1;}else
        {
            if (i_sm==global_N_smooth-1)  i2-=global_N_smooth-1;
            else i2 = i_sm+1;
        }
    */
    
    if (fraction_j <= 0) {
        if (j_sm==0)  j2=global_N_smooth-1;
        else j2 = j_sm-1;
    }else
    {
        if (j_sm==global_N_smooth-1)  j2=0;
        else j2 = j_sm+1;
    }
    
    if (fraction_i <= 0) {
        if (i_sm==0)  i2=global_N_smooth-1;
        else i2 = i_sm-1;}else
        {
            if (i_sm==global_N_smooth-1)  i2=0;
            else i2 = i_sm+1;
        }
    
    
    val = cell_c * box[(k_sm+global_N_smooth*(j_sm+global_N_smooth*i_sm))]
    + cell_x * box[(k_sm+global_N_smooth*(j_sm+global_N_smooth*i2))]
    + cell_y * box[(k_sm+global_N_smooth*(j2+global_N_smooth*i_sm))]
    + cell_xy * box[(k_sm+global_N_smooth*(j2+global_N_smooth*i2))];
    
   // if (j_sm == 0) printf("CiC function: %e \t %e \t %e \t %e \t %e \t %e \t %e \t %e \t %e \t %e \t %d\n",fraction_i,fraction_j,del_r,fraction_ii,fraction_jj,cell_c,cell_x,cell_y,cell_xy,val,j2);
    //TEST
    val = box[(k_sm+global_N_smooth*(j_sm+global_N_smooth*i_sm))];
    
    return val;
}


/******************************************************************************/
// get r(z) in Mpc/h

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
    
    //	for (int ii=0;ii<30000;ii++) printf("ii: %d \t DD: %f \t ZZ: %f \n",ii,x[ii],y[ii]);
    
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
    for (k=n-1;k>=anf;k--){
        y2[k]=y2[k]*y2[k+1]+u[k];
    }	
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
    
    for (int i=0;i<30000;i++)
    {
        ZZ[i]=i*0.001;
        DD[i]=get_int_r(ZZ[i]);
    }
    
    cubic_spline(DD,ZZ,1,30000,1.0e30,1.0e30,DD2);
    
}
