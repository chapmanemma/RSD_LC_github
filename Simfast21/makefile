#compiler
cc = /usr/local/bin/gcc-6
mpicc = /usr/local/bin/mpicc
#complilation flags
fftwf = -lfftw3f -lfftw3f_threads
fftwd = -lfftw3 -lfftw3_threads
omp = -fopenmp -D_OMPTHREAD_
gsl = -lgsl -lgslcblas
deb = -g
#flags = -std=c99 -Wall -O3 -march=native $(fftwd) $(fftwf) $(omp) $(gsl) -lm
#flags = -std=c99 -Wall -O3 $(fftwd) $(fftwf) $(omp) $(gsl) -lm
flags = -L/usr/local/lib -L/usr/local/Cellar/gcc/6.3.0_1/lib/gcc/6 -lgsl -lgslcblas -lfftw3f_omp -lfftw3f -lm -I/usr/local/include -I/usr/local/Cellar/gcc/6.3.0_1/lib/gcc/6/gcc/x86_64-apple-darwin15.6.0/6.3.0/include -lgsl -lgslcblas -lfftw3f_omp -lfftw3f -lm -std=c99 -Wall -O3 $(fftwd) $(fftwf) $(omp) $(gsl) -lm

aux = auxiliary.o Input_variables.o 


simfast21: get_densityfield get_velocityfield \
     get_halos_fcoll get_nldensity adjust_halos \
     get_HIIbubbles get_SFR \
     xalpha xc epsilonXon integratexe \
     integrateTempX t21 vel opdepth \
     lc RSD_LC

%.o: %.c auxiliary.h Input_variables.h
	${cc} ${flags} -c -o $@ $<

get_densityfield: $(aux) get_densityfield.o
	$(cc) -o get_densityfield get_densityfield.o $(aux) $(flags)

get_velocityfield: $(aux) get_velocityfield.o
	$(cc) -o get_velocityfield get_velocityfield.o $(aux) $(flags)

get_halos_fcoll: $(aux) get_halos_fcoll.o
	$(cc) -o get_halos_fcoll get_halos_fcoll.o $(aux) $(flags)

adjust_halos: $(aux) adjust_halos.o
	$(cc) -o adjust_halos adjust_halos.o $(aux) $(flags)

get_nldensity: $(aux) get_nldensity.o
	$(cc) -o get_nldensity get_nldensity.o $(aux) $(flags)

get_HIIbubbles: $(aux) get_HIIbubbles.o
	$(cc) -o get_HIIbubbles get_HIIbubbles.o $(aux) $(flags)

get_SFR: $(aux) get_SFR.o
	$(cc) -o get_SFR get_SFR.o $(aux) $(flags)

xalpha: $(aux) xalpha.o
	$(cc) -o xalpha xalpha.o $(aux) $(flags)

xc: $(aux) xc.o
	$(cc) -o xc xc.o $(aux) $(flags)

epsilonXon: $(aux) epsilonXon.o
	$(cc) -o epsilonXon epsilonXon.o $(aux) $(flags)

integratexe: $(aux) integratexe.o
	$(cc) -o integratexe integratexe.o $(aux) $(flags)

integrateTempX: $(aux) integrateTempX.o
	$(cc) -o integrateTempX integrateTempX.o $(aux) $(flags)

t21: $(aux) t21.o
	$(cc) -o t21 t21.o $(aux) $(flags)

RSD_LC: $(aux) RSD_LC_v1.10.o
	mpicc -o RSD_LC RSD_LC_v1.10.o $(aux) $(flags) -L/usr/local/Cellar/open-mpi/2.0.2/lib -I/usr/local/Cellar/open-mpi/2.0.2/include

vel: $(aux) vel.o
	$(cc) -o vel vel.o $(aux) $(flags)

lc: $(aux) lightcone_0.3.o
	$(cc) -o lc lightcone_0.3.o $(aux) $(flags)

opdepth: Input_variables.o opdepth.o
	$(cc) -o opdepth opdepth.o Input_variables.o -std=c99 -Wall -O3 -lm 

clean:
	rm *.o get_densityfield get_velocityfield \
                   get_halos_fcoll \
                   get_nldensity adjust_halos \
                   get_HIIbubbles\
                   get_SFR \
                   xalpha xc epsilonXon integratexe \
                   integrateTempX vel t21_vpec t21 opdepth \
		   lc RSD_LC
