PUBLIC asmInitInterleave

.code

asmInitInterleave PROC
    xor     eax, eax        ; zero register
    adox    rax, rax        ; OF = 0
    adcx    rax, rax        ; CF = 0
    ret
asmInitInterleave ENDP

END