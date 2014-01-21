uses Dos, Crt;
//{$Define PtrArr}
//{$ifdef PtrArr}
//type Iarr = array[word] of longint;
//     IntArr = ^IArr;
//{$else}
type IntArr = array of longint;
//{$endif}
const csize = 2000500;

procedure Test(src,dst:IntArr);
  var i,j:integer;
begin
  for i:=0 to 10000000000000000000 do
  begin
    for j:=0 to csize-1 do
    begin
      dst[j]:=src[j]*src[j];
    end;
  end;
end;

var Time:cardinal;
    //a,b:IntArr;
    a,b:array[2000500] of longint;
    i:integer;
    h, m, s1, s2, hund : Word;

begin
 //выделение памяти
 for i:=0 to csize-1 do
  a[i]:=i*2+1;
 getTime(h,m,s1,hund);
 Test(a,b);
 getTime(h,m,s2,hund);
 writeln(s1);
 writeln(s2);
 writeln(a[10]);
 writeln(b[10]);
 writeln(s2-s1);
 //удаление памяти
end.
