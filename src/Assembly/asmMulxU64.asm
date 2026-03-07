; asmMulxU64.asm 
; Implements the asm intrinsic call for mulx, to be used when Leibniz is compiled with MSVC

OPTION CASEMAP:NONE

.code

PUBLIC asmMulxU64

asmMulxU64 PROC
    ; Windows x64 ABI
    ; RCX = a
    ; RDX = b
    ; R8  = hi pointer

    mov rax, rcx        ; move a
    mov r9, rdx         ; copy b (mulx cannot use rdx directly)

    mulx r10, rax, r9   ; r10 = hi, rax = lo

    mov [r8], r10
    ret
asmMulxU64 ENDP

END