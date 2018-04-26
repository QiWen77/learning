with open('1') as f: 
    for line in f.readlines():
        st = str(line) 
        l1 = st.split('(')
        l1 = l1.split(')')
        l1 = l1.split(',')
        print l1
