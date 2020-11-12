C FILE: vvh_routines.f

        SUBROUTINE Tensor_Inner_Product_3X3(a,b,T)
            
            implicit none
            real(8), dimension(3,3), intent(in) :: a, b
            real(8), intent(out) :: T!, TX

            !TX = sum(sum(a*b,2))
            
            T = a(1,1)*b(1,1) + a(1,2)*b(1,2) + a(1,3)*b(1,3) 
            T = T + a(2,1)*b(2,1) + a(2,2)*b(2,2) + a(2,3)*b(2,3)
            T = T + a(3,1)*b(3,1) + a(3,2)*b(3,2) + a(3,3)*b(3,3)                

        END SUBROUTINE

        SUBROUTINE HENCKY_1D(mu, etr, Ei, dWtr, d2Wtr, energye)

            implicit none
            real(8), intent(in) :: mu, etr(3,1), Ei(3,3,3)
            real(8) :: G, eps(3), vdWtr(3)
            real(8), intent(out) :: dWtr(3,3), d2Wtr(3,3), energye

            eps(:) = 0.d0

            G = mu

            eps=[etr(1,1), etr(2,1), etr(3,1)]

            vdWtr(:) = 2*G*eps(:)

            dWtr(1, 1:3) = [vdWtr(1),    0.0d0,    0.0d0]
            dWtr(2, 1:3) = [   0.0d0, vdWtr(2),    0.0d0]
            dWtr(3, 1:3) = [   0.0d0,    0.0d0, vdWtr(3)]

            d2Wtr(1, 1:3) = [2*G,    0.0d0,    0.0d0]
            d2Wtr(2, 1:3) = [   0.0d0, 2*G,    0.0d0]
            d2Wtr(3, 1:3) = [   0.0d0,    0.0d0, 2*G]

            energye = G*(eps(1)**2 + eps(2)**2 + eps(3)**2)

        END SUBROUTINE

        SUBROUTINE Hencky_3D(mu, etr, Ei, dWtr, d2Wtr, energye)
            
            implicit none
            real(8), intent(in) :: mu, etr(3,3), Ei(3,3,3)
            real(8) :: G, eps(3), vdWtr(3)
            real(8), intent(out) :: dWtr(3,3), d2Wtr(3,3), energye

            eps(:) = 0.d0

            G = mu

            call Tensor_Inner_Product_3X3(etr, Ei(:,:,1), eps(1))
            call Tensor_Inner_Product_3X3(etr, Ei(:,:,2), eps(2))
            call Tensor_Inner_Product_3X3(etr, Ei(:,:,3), eps(3))

            vdWtr(:) = 2*G*eps(:)

            dWtr(1, 1:3) = [vdWtr(1),    0.0d0,    0.0d0]
            dWtr(2, 1:3) = [   0.0d0, vdWtr(2),    0.0d0]
            dWtr(3, 1:3) = [   0.0d0,    0.0d0, vdWtr(3)]

            d2Wtr(1, 1:3) = [2*G,    0.0d0,    0.0d0]
            d2Wtr(2, 1:3) = [   0.0d0, 2*G,    0.0d0]
            d2Wtr(3, 1:3) = [   0.0d0,    0.0d0, 2*G]

            energye = G*(eps(1)**2 + eps(2)**2 + eps(3)**2)

        END SUBROUTINE

C END FILE