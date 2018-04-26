for d in ./launcher*; do
  echo "$d"
  if [ -d $d ];then
  cd $d
  else
  continue
  fi
  cd static
  #cd bandstructure
  #rm -rf edos
  #mkdir edos
  #cd edos
  #mv edos.txt element.txt
  #cp ../vasprun.xml .
  #cp ../POSCAR .
  #sed 's/要被取代的字串/新的字串/g' 
  #sed 's/[ ][ ]*/,/g' element.txt
  nl POSCAR | sed -n '1p' >mp222.txt
  awk '{print $2,$3,$4,$5,$6,$7}' mp222.txt> fermi.txt
  sed 's/ //g' fermi.txt>formula.txt
  #rm -rf  element.txt mp111.txt
  
  read -r formula<formula.txt
  #read -r element<element.txt
  echo "$formula"
  sed -n '6,7'p  POSCAR
  #source /lustre/home/acct-umjzhh/umjzhh-2/qiwen/monodoping_MAPbI3/kit.sh
  #command  $kit 
  dopant_orig=$(grep 'ZVAL'  POTCAR   |  sed -n '1'p |awk '{print $6}')
  echo "dopant_orig:$dopant_orig"
  cp ../../integrated_dos_number.py   .
  python integrated_dos_number.py
  d_outcar=$(sed -n '/# of ion/,/2/p'   OUTCAR  | sed -n '3p'|awk '{print $4}')
  echo "d_outcar:$d_outcar"
  total_outcar=$(sed -n '/# of ion/,/2/p'   OUTCAR  | sed -n '3p'|awk '{print $5}')
  echo "total_outcar:$total_outcar"
  sed -n '/# of ion/,/2/p'   OUTCAR  | sed -n '3p'
  #mkdir ../../entry/"$formula"
  #cp *   ../../entry/"$formula"
  #grep 'energy  without entropy' OUTCAR
  #echo "$element"
  #split_dos>fermi.txt
  #cat -b fermi.txt>>"$formula".txt
  #mv DOSCAR _DOSCAR
  #rm DOS*
  #mv _DOSCAR DOSCAR
  rm *.txt
  #cp ~/qiwen/structure_analyzer_B222.py  .
  #python structure_analyzer_B222.py 
  #python ../../cbmvbm.py  
  #pmg plotdos vasprun.xml -e $element -f "$formula".png #&& cp -r ./*.png ../..\/../EDOS
  #rm vasprun.xml POSCAR
  cd ../..
done
