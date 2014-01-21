	.file	"3.c"
	.section	.text.startup,"ax",@progbits
	.p2align 4,,15
	.globl	main
	.type	main, @function
main:
.LFB0:
	.cfi_startproc
	xorpd	%xmm0, %xmm0
	movl	$temp1.1580+262144, %ecx
	movl	$temp1.1580+33816576, %esi
.L2:
	leaq	-262144(%rcx), %rax
	.p2align 4,,10
	.p2align 3
.L6:
	leaq	2048(%rax), %rdx
	.p2align 4,,10
	.p2align 3
.L3:
	movapd	%xmm0, (%rax)
	addq	$16, %rax
	cmpq	%rdx, %rax
	jne	.L3
	cmpq	%rcx, %rax
	jne	.L6
	leaq	262144(%rax), %rcx
	cmpq	%rsi, %rcx
	jne	.L2
	xorl	%eax, %eax
	ret
	.cfi_endproc
.LFE0:
	.size	main, .-main
	.local	temp1.1580
	.comm	temp1.1580,67108864,32
	.ident	"GCC: (Ubuntu/Linaro 4.7.3-1ubuntu1) 4.7.3"
	.section	.note.GNU-stack,"",@progbits
